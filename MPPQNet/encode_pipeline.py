import argparse
import os
import glob
import struct
import numpy as np
import torch
import open3d as o3d
from feature_extraction import build_feature_matrix
from mpnn_qnn import run_mppn
from networks import PNetwork, FNetwork
from tqdm import tqdm

def encode_one(ply_path, pf_weight, K, num_bins, out_dir, status_bar):
    base = os.path.splitext(os.path.basename(ply_path))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入預訓練的 PNetwork 與 FNetwork
    status_bar.set_description(f"[{base}] 載入 MPNN 權重")
    status_bar.refresh()
    pnet = PNetwork(input_dim=num_bins, output_dim=K).to(device)
    fnet = FNetwork(input_dim=num_bins, output_dim=K).to(device)
    ckpt = torch.load(pf_weight, map_location=device)
    pnet.load_state_dict(ckpt['pnet'])
    fnet.load_state_dict(ckpt['fnet'])
    pnet.eval(); fnet.eval()

    # 2. 讀取 .ply，建立 6 維特徵矩陣 F (M×6)
    status_bar.set_description(f"[{base}] 建立 6D 特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    coords = (pts - mn) / (mx - mn)          # 座標歸一化至 [0,1]
    colors = np.asarray(pcd.colors) * 255.0  # 顏色放大至 [0,255]
    F = build_feature_matrix(coords, colors) # shape (M,6)
    M, D = F.shape                           # D==6

    # 3. Stage1：對每維 histogram + MPNN 量化 → thresholds, centers, labels1
    status_bar.set_description(f"[{base}] Stage1 MPNN 量化")
    status_bar.refresh()
    thr1_list, cen1_list, lab1_list = [], [], []
    for d in range(D):
        # 3.1 計算第 d 維的 histogram 並正規化為機率分布
        hist, _ = np.histogram(
            F[:, d], bins=num_bins,
            range=(F[:, d].min(), F[:, d].max())
        )
        hist = hist.astype(np.float32)
        hist /= hist.sum()

        # 3.2 用 run_mppn 產生 f_vec 與 thresholds
        _, f_vec, thr = run_mppn(
            hist, K,
            device=device,
            pf_weight=pf_weight
        )
        centers = np.sort(f_vec).astype(np.float32)

        thr1_list.append(thr.astype(np.float32))
        cen1_list.append(centers)
        # 3.3 逐點量化得到 labels1（0..K-1）
        lab1_list.append(np.digitize(F[:, d], thr))

    thr1_arr = np.stack(thr1_list, axis=0)
    cen1_arr = np.stack(cen1_list, axis=0)
    labs1    = np.stack(lab1_list, axis=1)

    # 4. 第一階段重建與殘差
    F_hat1 = cen1_arr[np.arange(D)[None, :], labs1]
    R      = F - F_hat1

    # 5. Stage2：對殘差 R 做 Min–Max 歸一化，再量化 → thr2,cen2,labs2
    status_bar.set_description(f"[{base}] Stage2 殘差量化")
    status_bar.refresh()
    thr2_list, cen2_list, lab2_list = [], [], []
    for d in range(D):
        # 5.1 取出原始殘差，算出 min/max
        R_d = R[:, d]  
        R_min, R_max = R_d.min(), R_d.max()
        # 5.2 歸一化到 [0,1]
        R_norm = (R_d - R_min) / (R_max - R_min)

        # 5.3 在歸一化後的 domain 上算 histogram & MPPN
        hist_r, _ = np.histogram(
            R_norm, bins=num_bins, range=(0.0, 1.0)
        )
        hist_r = hist_r.astype(np.float32)
        hist_r /= hist_r.sum()

        _, f_vec2_norm, thr2_norm = run_mppn(
            hist_r, K,
            device=device,
            pf_weight=pf_weight
        )
        centers2_norm = np.sort(f_vec2_norm).astype(np.float32)

        # 5.4 把 thresholds/centers 從 [0,1] 映回原始 residual domain
        thr2      = thr2_norm * (R_max - R_min) + R_min
        centers2  = centers2_norm * (R_max - R_min) + R_min

        thr2_list.append(thr2.astype(np.float32))
        cen2_list.append(centers2)
        # 5.5 用回到原 domain 的 thresholds 分類
        lab2_list.append(np.digitize(R_d, thr2))

    thr2_arr = np.stack(thr2_list, axis=0)
    cen2_arr = np.stack(cen2_list, axis=0)
    labs2    = np.stack(lab2_list, axis=1)

    # 6. 計算 Gaussian 參數 μ, σ
    status_bar.set_description(f"[{base}] 計算 Gaussian 參數")
    status_bar.refresh()
    mu1_arr    = np.zeros((D, K), dtype=np.float32)
    sigma1_arr = np.zeros((D, K), dtype=np.float32)
    mu2_arr    = np.zeros((D, K), dtype=np.float32)
    sigma2_arr = np.zeros((D, K), dtype=np.float32)
    for d in range(D):
        for k in range(K):
            mask1 = (labs1[:, d] == k)
            if mask1.any():
                vals1 = F[mask1, d]
                mu1_arr[d, k]    = vals1.mean()
                sigma1_arr[d, k] = ((vals1 - mu1_arr[d, k])**2).mean()
            mask2 = (labs2[:, d] == k)
            if mask2.any():
                vals2 = R[mask2, d]
                mu2_arr[d, k]    = vals2.mean()
                sigma2_arr[d, k] = ((vals2 - mu2_arr[d, k])**2).mean()

    # 7. 序列化 bitstream (.bin)
    status_bar.set_description(f"[{base}] 序列化 bitstream")
    status_bar.refresh()
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(bin_path, 'wb') as fb:
        # Header: num_bins, K, D, M
        fb.write(struct.pack('<IIII', num_bins, K, D, M))
        # 寫入 mn、mx
        fb.write(mn.astype(np.float32).tobytes())
        fb.write(mx.astype(np.float32).tobytes())
        # Stage1: thresholds, centers, μ, σ
        fb.write(thr1_arr.tobytes())
        fb.write(cen1_arr.tobytes())
        fb.write(mu1_arr.tobytes())
        fb.write(sigma1_arr.tobytes())
        # Stage2: thresholds, centers, μ, σ
        fb.write(thr2_arr.tobytes())
        fb.write(cen2_arr.tobytes())
        fb.write(mu2_arr.tobytes())
        fb.write(sigma2_arr.tobytes())
        # Point-level labels1, labels2
        fb.write(labs1.astype(np.uint8).tobytes())
        fb.write(labs2.astype(np.uint8).tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="MPNN-based 3D 點雲編碼 → bitstream"
    )
    parser.add_argument("--input_dir", required=True,
                        help="原始 .ply 資料夾")
    parser.add_argument("--pf_weight", required=True,
                        help="MPPN PF 權重檔 (.pth)")
    parser.add_argument("--K", type=int, default=6,
                        help="量化層級數 K")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="Histogram bins 數")
    parser.add_argument("--out_dir", default="bitstreams",
                        help="輸出 bitstream 資料夾")
    args = parser.parse_args()

    ply_list  = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    outer_bar = tqdm(ply_list, desc="Encoding pipeline",
                     unit="file", position=0)
    status    = tqdm(total=0, bar_format="{desc}",
                     position=1, leave=True)
    for ply in outer_bar:
        encode_one(ply, args.pf_weight, args.K,
                   args.num_bins, args.out_dir, status)
        outer_bar.update(0)
    status.close()
    print("Encode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
