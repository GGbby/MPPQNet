# Updated pipeline_kmeans.py with 坐標歸一化到 [0,1]

import argparse
import os
import glob
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn import run_mppn
from dictionary_and_gaussian import compute_gaussian_params_gpu


def reconstruct_F_from_fvecs_and_thresholds(F, f_vecs, thresholds_list):
    """
    用硬編碼方式將原始 F (M×N_dims) 依 f_vecs / thresholds 重建成 F_hat。
    """
    M, N_dims = F.shape
    F_hat = np.zeros_like(F)
    for i in range(N_dims):
        bins = np.digitize(F[:, i], thresholds_list[i])
        F_hat[:, i] = f_vecs[i][bins]
    return F_hat

def main(input_dir, pf_weight=None):
    # 設定裝置 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置：", device)

    # 搜尋所有 .ply 檔
    ply_list = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    if not ply_list:
        print(f"找不到任何 .ply 檔於 {input_dir}")
        return

    # 建立儲存資料夾
    npz_dir = os.path.join("results", "npz")
    ply_dir = os.path.join("results", "ply")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # 批次處理所有 .ply，顯示進度
    for ply_path in tqdm(ply_list, desc="KMeans pipeline", unit="file"):
        base = os.path.splitext(os.path.basename(ply_path))[0]

        # 1. 讀取點雲並將座標歸一化到 [0,1]
        pcd = o3d.io.read_point_cloud(ply_path)
        pc_orig = np.asarray(pcd.points)  # shape (M, 3)
        min_coords = pc_orig.min(axis=0)
        max_coords = pc_orig.max(axis=0)
        scale = max_coords - min_coords
        coords_norm = (pc_orig - min_coords) / scale  # 歸一化

        # 2. 讀取顏色，並建構 6 維特徵矩陣
        colors = np.asarray(pcd.colors)
        F = build_feature_matrix(coords_norm, colors)  # shape (M, 6)

        # 3. 計算每維特徵的直方圖
        num_bins = 64
        hists = compute_histogram_per_column(F, num_bins)  # list of 6 histograms

        # 4. MPPN forward 取得 f_vec 與 thresholds
        K_mppn = 6
        all_f, thresholds_list = [], []
        for H in hists:
            _, f_vec, thr = run_mppn(
                H, K_mppn,
                epochs=1, lr=1e-3,
                device=device,
                pf_weight=pf_weight
            )
            all_f.append(f_vec)
            thresholds_list.append(thr)
        all_feats = np.vstack(all_f)  # shape (N_dims, K)

        # 5. 對「每個點」的 6D 特徵 F (M×6) 做 KMeans 分群
        num_clusters = 128
        labels, centers = kmeans_gpu(F, num_clusters, iters=50, device=device)

        # 6. 直接用分群結果還原：每個點的新特徵 = centers[labels[p]]
        F_hat = centers[labels]  # shape (M,6)

        # 7. 儲存 .npz
        np.savez(
            os.path.join(npz_dir, f"{base}_kmeans.npz"),
            all_feats=all_feats,
            all_feats_hat=F_hat,
            F_original=F,
            F_hat=F_hat
        )

        # 8. 寫出重建後的點雲 (包含歸一化後的座標及顏色)
        coords_hat = F_hat[:, :3]  # 已在 [0,1] 範圍內
        colors_hat = F_hat[:, 3:] / 255.0
        recon = o3d.geometry.PointCloud()
        recon.points = o3d.utility.Vector3dVector(coords_hat)
        recon.colors = o3d.utility.Vector3dVector(colors_hat)
        o3d.io.write_point_cloud(
            os.path.join(ply_dir, f"{base}_kmeans.ply"),
            recon
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch GPU KMeans pipeline")
    parser.add_argument("--input_dir", required=True, help="資料夾，含多個 .ply 檔")
    parser.add_argument("--pf", default=None, help="可選：mppn_pf.pth 權重檔")
    args = parser.parse_args()
    main(args.input_dir, args.pf)
