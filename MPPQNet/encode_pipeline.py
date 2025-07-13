# encode_pipeline_v2_bin.py
# 將 .ply 轉為 .bin bit-stream，支援 PF soft-VQ、雙層 Gaussian 殘差處理，並可選擇只壓縮最新檔案

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
    ply_path, pf_weight, K, num_bins,
    num_clusters, num_res_clusters,
    out_dir, status_bar, alpha=0.7
):
    """
    將單一 .ply 檔編碼為 .bin bit-stream。
    步驟：
      1. 讀取點雲並建構 6 維特徵矩陣 (XYZ 归一化 + RGB)
      2. PFNetwork 推理 -> pf_codewords + thresholds
      3. Soft-VQ 軟分配重構
      4. Gaussian μ + cluster centers 混合
      5. Coarse quantization (KMeans)
      6. Residual quantization (KMeans + Gaussian)
      7. pickle 序列化輸出 .bin
    """
    base = os.path.splitext(os.path.basename(ply_path))[0]

    # 1. 建立特徵矩陣
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points, dtype=np.float32)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - min_c) / (max_c - min_c + 1e-8)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # (N,6)

    # 2. PFNetwork 推理
    status_bar.set_description(f"[{base}] PF 推理 (K={K}, bins={num_bins})")
    status_bar.refresh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_f, thr_list = [], []
    for H in compute_histogram_per_column(F, num_bins):
        _, f_vec, thr = run_mppn(H, K=K, epochs=1, device=device, pf_weight=pf_weight)
        all_f.append(f_vec)
        thr_list.append(thr)
    pf_codewords = np.vstack(all_f)  # (6, K)

    # 3. Soft-VQ 軟分配
    labels_pf, centers_pf = kmeans_gpu(pf_codewords, num_clusters, device=device)
    centers_t = torch.from_numpy(centers_pf).to(device)
    feats_t = torch.from_numpy(pf_codewords).to(device)
    dists = torch.cdist(feats_t, centers_t)               # L2 距離
    weights = torch.softmax(-dists, dim=1)                # 軟分配權重
    soft_fvecs = (weights @ centers_t).cpu().numpy()      # 重構特徵

    # 4. Gaussian μ + center 混合
    omegas_pf = compute_gaussian_params_gpu(pf_codewords, labels_pf, centers_pf)
    mu_pf = np.stack([o[0] for o in omegas_pf], axis=0)
    sigma_pf = np.stack([o[1] for o in omegas_pf], axis=0)
    mixed_pf = alpha * mu_pf + (1 - alpha) * centers_pf

    # 5. 第一階段 Coarse quantization
    status_bar.set_description(f"[{base}] 第一階段分群 (Coarse: {num_clusters})")
    status_bar.refresh()
    labels_pt, centers_pt = kmeans_gpu(F, num_clusters, device=device)

    # 6. 第二階段 Residual quantization
    status_bar.set_description(f"[{base}] 第二階段分群 (Residual: {num_res_clusters})")
    status_bar.refresh()
    residuals = F - centers_pt[labels_pt]
    labels_res, centers_res = kmeans_gpu(residuals, num_res_clusters, device=device)
    omegas_res = compute_gaussian_params_gpu(residuals, labels_res, centers_res)
    mu_res = np.stack([o[0] for o in omegas_res], axis=0)
    sigma_res = np.stack([o[1] for o in omegas_res], axis=0)

    # 7. 序列化輸出 .bin
    status_bar.set_description(f"[{base}] 寫出 bit-stream (.bin)")
    status_bar.refresh()
    payload = {
        'pf_codewords':    soft_fvecs,
        'pf_thr':          thr_list,
        'gauss_mu_pf':     mixed_pf,
        'gauss_sigma_pf':  sigma_pf,
        'pt_labels':       labels_pt,
        'pt_centers':      centers_pt,
        'res_labels':      labels_res,
        'res_centers':     centers_res,
        'gauss_mu_res':    mu_res,
        'gauss_sigma_res': sigma_res
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(
        description="Encode .ply -> .bin bit-stream with PF soft-VQ + Gaussian"
    )
    parser.add_argument("--input_dir",       required=True, help="來源 .ply 資料夾")
    parser.add_argument("--pf",              required=True, help="PFNetwork 權重檔 (.pth)")
    parser.add_argument("--K",   type=int,   default=6,    help="PFNetwork 輸出維度 K")
    parser.add_argument("--num_bins",        type=int,   default=64,   help="histogram bin 數")
    parser.add_argument("--num_clusters",    type=int,   default=128,  help="第一階段 cluster 數")
    parser.add_argument("--num_res_clusters",type=int,   default=64,   help="殘差 cluster 數")
    parser.add_argument("--out_dir",         default="bitstreams_bin", help="輸出 .bin 資料夾")
    parser.add_argument("--sample_n",        type=int,   default=None, help="隨機抽樣處理數量")
    parser.add_argument("--latest",          action="store_true",      help="只處理最後修改的 .ply")
    parser.add_argument("--alpha",           type=float, default=0.7,  help="Gaussian μ + center 混合比例")
    args = parser.parse_args()

    # 列出並排序所有 .ply
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")),
                   key=os.path.getmtime)

    # 處理邏輯：latest > sample_n > 全部
    if args.latest and files:
        files = [files[-1]]
    elif args.sample_n and args.sample_n < len(files):
        files = random.sample(files, args.sample_n)

    outer = tqdm(files, desc="Encoding pipeline", unit="file")
    status = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for ply in outer:
        encode_one(
            ply, args.pf, args.K, args.num_bins,
            args.num_clusters, args.num_res_clusters,
            args.out_dir, status, args.alpha
        )
        outer.update(0)
    status.close()
    print("Encode 完成 ▶", args.out_dir)


if __name__ == "__main__":
    main()
