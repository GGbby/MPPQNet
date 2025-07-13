# encode_pipeline_v2_bin.py
# 針對邊緣設備/process: 將 .ply 轉為 .bin bit-stream
# 支持動態 K、Soft-Assignment VQ 以及 Gaussian μ + Center 混合
# 序列化使用 Python pickle 將 dict dump 為二進位檔

import argparse
import os
import glob
import random
import pickle
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn import run_mppn
from dictionary_and_gaussian import compute_gaussian_params_gpu
from compression_utils import kmeans_gpu


def encode_one(
    ply_path,
    pf_weight,
    K,
    num_bins,
    num_clusters,
    num_res_clusters,
    out_dir,
    status_bar,
    alpha=0.7
):
    """
    將單一 .ply 檔案壓縮並輸出為 .bin bit-stream：
    1. 建立六維特徵 (XYZ + RGB)
    2. PFNetwork 推理 → pf_codewords + thresholds
    3. Soft-VQ 重構特徵向量
    4. Gaussian μ + center 混合
    5. Coarse quantization (KMeans)
    6. Residual quantization (KMeans + Gaussian)
    7. pickle.dump 為 .bin
    """
    base = os.path.splitext(os.path.basename(ply_path))[0]

    # 1. 讀取點雲並建特徵矩陣
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points).astype(np.float32)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - min_c) / (max_c - min_c)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # M×6

    # 2. PF 推理
    status_bar.set_description(f"[{base}] PF 推理 (K={K}, bins={num_bins})")
    status_bar.refresh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_f, thr_list = [], []
    for H in compute_histogram_per_column(F, num_bins):
        p_vec, f_vec, thr = run_mppn(
            H, K=K, epochs=1, device=device, pf_weight=pf_weight
        )
        all_f.append(f_vec)
        thr_list.append(thr)
    pf_codewords = np.vstack(all_f)  # 6×K

    # 3. Soft-Assignment VQ
    labels_pf, centers_pf = kmeans_gpu(pf_codewords, num_clusters, device=device)
    centers_t = torch.from_numpy(centers_pf).to(device)
    feats_t = torch.from_numpy(pf_codewords).to(device)
    dists = torch.cdist(feats_t, centers_t)
    weights = torch.softmax(-dists, dim=1)
    soft_fvecs = torch.matmul(weights, centers_t).cpu().numpy()

    # 4. Gaussian μ + Center 混合
    omegas_pf = compute_gaussian_params_gpu(pf_codewords, labels_pf, centers_pf)
    mu_pf = np.stack([o[0] for o in omegas_pf], axis=0)
    sigma_pf = np.stack([o[1] for o in omegas_pf], axis=0)
    mixed_pf = alpha * mu_pf + (1 - alpha) * centers_pf

    # 5. Coarse quantization
    status_bar.set_description(f"[{base}] Coarse quantization ({num_clusters})")
    status_bar.refresh()
    labels_pt, centers_pt = kmeans_gpu(F, num_clusters, device=device)

    # 6. Residual quantization
    status_bar.set_description(f"[{base}] Residual quantization ({num_res_clusters})")
    status_bar.refresh()
    residuals = F - centers_pt[labels_pt]
    labels_res, centers_res = kmeans_gpu(residuals, num_res_clusters, device=device)
    omegas_res = compute_gaussian_params_gpu(residuals, labels_res, centers_res)
    mu_res = np.stack([o[0] for o in omegas_res], axis=0)
    sigma_res = np.stack([o[1] for o in omegas_res], axis=0)

    # 7. 輸出 .bin via pickle
    status_bar.set_description(f"[{base}] 序列化 bit-stream (.bin)")
    status_bar.refresh()
    payload = {
        'pf_codewords': soft_fvecs,
        'pf_thr':       thr_list,
        'gauss_mu_pf':  mixed_pf,
        'gauss_sigma_pf': sigma_pf,
        'pt_labels':    labels_pt,
        'pt_centers':   centers_pt,
        'res_labels':   labels_res,
        'res_centers':  centers_res,
        'gauss_mu_res': mu_res,
        'gauss_sigma_res': sigma_res
    }
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(bin_path, 'wb') as fb:
        pickle.dump(payload, fb)


def main():
    parser = argparse.ArgumentParser(
        description="Encode .ply to .bin bit-stream (PF soft-VQ + Gaussian)"
    )
    parser.add_argument("--input_dir", required=True, help="原始 .ply 資料夾")
    parser.add_argument("--pf",        required=True, help="mppn_pf.pth 權重檔")
    parser.add_argument("--K",  type=int, default=6,  help="PFNetwork 輸出 K 值")
    parser.add_argument("--num_bins", type=int, default=64,  help="直方圖 bins")
    parser.add_argument("--num_clusters",     type=int, default=128, help="coarse cluster")
    parser.add_argument("--num_res_clusters", type=int, default=64,  help="residual cluster")
    parser.add_argument("--out_dir", default="bitstreams_bin", help="輸出 .bin 資料夾")
    parser.add_argument("--sample_n", type=int, default=None, help="隨機抽樣 .ply 數量")
    parser.add_argument("--alpha", type=float, default=0.7, help="Gaussian μ + center 混合比例")
    args = parser.parse_args()

    ply_list = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    if args.sample_n and args.sample_n < len(ply_list):
        ply_list = random.sample(ply_list, args.sample_n)

    outer = tqdm(ply_list, desc="Encoding pipeline", unit="file")
    status = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for ply in outer:
        encode_one(
            ply,
            args.pf,
            args.K,
            args.num_bins,
            args.num_clusters,
            args.num_res_clusters,
            args.out_dir,
            status,
            alpha=args.alpha
        )
        outer.update(0)
    status.close()
    print("Encode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
