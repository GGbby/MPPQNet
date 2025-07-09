# encode_pipeline.py (v2)
# 在原始腳本基礎上新增：
#  - 支持動態 K 值參數化
#  - Soft-Assignment Vector Quantization (Soft-VQ)
#  - Gaussian μ 與 Center 混合重建 (alpha 可調)
# 保留原始註解與參數設定，僅在 PF 階段與 bitstream 輸出混入優化策略

import argparse
import os
import glob
import random
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
    對單一 .ply 檔案執行兩階段編碼：
    1. 建立六維特徵 (坐標 XYZ + 顏色 RGB)
    2. PFNetwork 推理 (MPPN) → pf_codewords + thresholds
    3. Soft-VQ 重構特徵向量 (soft assignment)
    4. Gaussian μ 與 KMeans center 混合 (alpha)
    5. 第一階段 KMeans coarse quantization
    6. 殘差計算 + 第二階段 KMeans residual quantization
    7. 序列化 .npz bit-stream (包含所有必要資料)
    """
    base = os.path.splitext(os.path.basename(ply_path))[0]

    # 1. 讀取點雲並建特徵矩陣
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points).astype(np.float32)
    # 座標正規化到 [0,1]
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - min_c) / (max_c - min_c)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # (M,6)

    # 2. PF 推理 (使用預訓練權重進行 single-pass forward)
    status_bar.set_description(f"[{base}] PF 推理 (K={K}, bins={num_bins})")
    status_bar.refresh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_f, thr_list = [], []
    for H in compute_histogram_per_column(F, num_bins):
        # 傳入動態 K
        p_vec, f_vec, thr = run_mppn(
            H, K=K, epochs=1, device=device, pf_weight=pf_weight
        )
        all_f.append(f_vec)
        thr_list.append(thr)
    pf_codewords = np.vstack(all_f)  # (6, K)

    # 3. Soft-Assignment Vector Quantization
    labels_pf, centers_pf = kmeans_gpu(pf_codewords, num_clusters, device=device)
    centers_t = torch.from_numpy(centers_pf).to(device)  # (C, D)
    feats_t = torch.from_numpy(pf_codewords).to(device)  # (N, D)
    dists = torch.cdist(feats_t, centers_t)              # (N, C)
    weights = torch.softmax(-dists, dim=1)               # soft assignments
    soft_fvecs = torch.matmul(weights, centers_t).cpu().numpy()  # (N, D)

    # 4. Gaussian μ 與 Center 混合重建
    omegas_pf = compute_gaussian_params_gpu(pf_codewords, labels_pf, centers_pf)
    mu_pf = np.stack([o[0] for o in omegas_pf], axis=0)         # (C, D)
    sigma_pf = np.stack([o[1] for o in omegas_pf], axis=0)      # (C, D, D)
    mixed_pf = alpha * mu_pf + (1 - alpha) * centers_pf         # (C, D)

    # 5. 第一階段 coarse quantization
    status_bar.set_description(f"[{base}] 第一階段分群 ({num_clusters})")
    status_bar.refresh()
    labels_pt, centers_pt = kmeans_gpu(F, num_clusters, device=device)

    # 6. 計算殘差並進行 residual quantization
    status_bar.set_description(f"[{base}] 殘差計算")
    status_bar.refresh()
    residuals = F - centers_pt[labels_pt]  # (M,6)
    status_bar.set_description(f"[{base}] 第二階段分群 ({num_res_clusters})")
    status_bar.refresh()
    labels_res, centers_res = kmeans_gpu(residuals, num_res_clusters, device=device)
    omegas_res = compute_gaussian_params_gpu(residuals, labels_res, centers_res)
    mu_res = np.stack([o[0] for o in omegas_res], axis=0)        # (C_res, D)
    sigma_res = np.stack([o[1] for o in omegas_res], axis=0)     # (C_res, D, D)

    # 7. 序列化 bit-stream (.npz)
    status_bar.set_description(f"[{base}] 序列化 bit-stream")
    status_bar.refresh()
    np.savez(
        os.path.join(out_dir, f"{base}_stream.npz"),
        # 使用 soft-VQ 重構的 pf_codewords
        pf_codewords=soft_fvecs,
        pf_thr=thr_list,
        gauss_mu_pf=mixed_pf,
        gauss_sigma_pf=sigma_pf,
        pt_labels=labels_pt,
        pt_centers=centers_pt,
        res_labels=labels_res,
        res_centers=centers_res,
        gauss_mu_res=mu_res,
        gauss_sigma_res=sigma_res
    )


def main():
    parser = argparse.ArgumentParser(
        description="Encode with PFNetwork (dynamic K), Soft-VQ + Gaussian mixing"
    )
    parser.add_argument(
        "--input_dir", required=True, help="原始 .ply 資料夾"
    )
    parser.add_argument(
        "--pf", required=True, help="mppn_pf.pth 權重檔"
    )
    parser.add_argument(
        "--K", type=int, default=6, help="PFNetwork 輸出維度 K 與 train pth 對應"
    )
    parser.add_argument(
        "--num_bins", type=int, default=64, help="直方圖 bins 數"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=128, help="第一階段 cluster 數"
    )
    parser.add_argument(
        "--num_res_clusters", type=int, default=64, help="殘差 cluster 數"
    )
    parser.add_argument(
        "--out_dir", default="bitstreams", help="輸出 .npz 資料夾"
    )
    parser.add_argument(
        "--sample_n", type=int, default=None, help="隨機抽取處理的 ply 檔數量"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Gaussian μ 與 Center 混合比例"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ply_list = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    if args.sample_n and 0 < args.sample_n < len(ply_list):
        ply_list = random.sample(ply_list, args.sample_n)

    outer_bar = tqdm(ply_list, desc="Encoding pipeline", unit="file")
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for ply in outer_bar:
        encode_one(
            ply, args.pf, args.K, args.num_bins,
            args.num_clusters, args.num_res_clusters,
            args.out_dir, status_bar, alpha=args.alpha
        )
        outer_bar.update(0)
    status_bar.close()
    print("Encode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
