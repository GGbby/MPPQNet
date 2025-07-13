import argparse
import os
import glob
import random
import pickle

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from feature_extraction import build_feature_matrix
from mpnn_qnn import run_mppn

def encode_one(ply_path, pf_weight, K, num_bins, out_dir, status_bar):
    base = os.path.splitext(os.path.basename(ply_path))[0]

    # 1. 讀取點雲，建 6D 特徵矩陣 (XYZ 归一化 + RGB)
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points, dtype=np.float32)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - min_c) / (max_c - min_c + 1e-8)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # shape: (N,6)

    # 2. GPU 上計算每維直方圖並呼叫 PFNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status_bar.set_description(f"[{base}] 直方圖 + PFNetwork 推理")
    status_bar.refresh()

    F_t = torch.from_numpy(F).to(device)  # (N,6)
    all_fvecs = []     # List of f_vec per dimension
    all_thresholds = []  # List of thr per dimension

    for d in range(F_t.shape[1]):
        col = F_t[:, d]
        # GPU histogram
        h = torch.histc(col, bins=num_bins, min=col.min().item(), max=col.max().item())
        h = h / h.sum()
        H = h.cpu().numpy().astype(np.float32)  # PFNetwork 輸入

        # run MPPN (PFNetwork + FNetwork)
        p_vec, f_vec, thr = run_mppn(
            H, K=K, epochs=1, device=device, pf_weight=pf_weight
        )
        all_fvecs.append(f_vec)    # numpy (K,)
        all_thresholds.append(thr)  # numpy (K-1,)

    fvecs_arr = np.stack(all_fvecs, axis=0)        # (6, K)
    thr_arr   = np.stack(all_thresholds, axis=0)   # (6, K-1)

    # 3. 量化編碼：利用 thresholds 做 bucketize，再用 fvecs 重建
    status_bar.set_description(f"[{base}] 量化編碼")
    status_bar.refresh()
    labels = torch.zeros_like(F_t, dtype=torch.long)
    F_hat = torch.zeros_like(F_t)

    for d in range(F_t.shape[1]):
        thr_t = torch.from_numpy(thr_arr[d]).to(device)  # (K-1,)
        # bucketize: labels 0..K-1
        idx = torch.bucketize(F_t[:, d], thr_t)
        labels[:, d] = idx
        fvals = torch.from_numpy(fvecs_arr[d]).to(device)  # (K,)
        F_hat[:, d] = fvals[idx]

    labels_np = labels.cpu().numpy()    # (N,6)
    F_hat_np  = F_hat.cpu().numpy()     # (N,6)

    # 4. 序列化輸出
    status_bar.set_description(f"[{base}] 寫出 .bin")
    status_bar.refresh()
    payload = {
        'shape':        F.shape,
        'fvecs':        fvecs_arr,
        'thresholds':   thr_arr,
        'labels':       labels_np,
        'recon_feats':  F_hat_np
    }
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(bin_path, 'wb') as f:
        pickle.dump(payload, f)

def main():
    parser = argparse.ArgumentParser(
        description="Encode .ply -> .bin using PFNetwork on GPU histc"
    )
    parser.add_argument("--input_dir", required=True, help="來源 .ply 資料夾")
    parser.add_argument("--pf",        required=True, help="PFNetwork 權重檔 (.pth)")
    parser.add_argument("--K",   type=int, default=6,  help="PFNetwork 類別數 K")
    parser.add_argument("--num_bins", type=int, default=64, help="直方圖 bins")
    parser.add_argument("--out_dir",   default="bitstreams_bin", help="輸出 .bin 資料夾")
    parser.add_argument("--latest",    action="store_true", help="只處理最後修改的 .ply")
    parser.add_argument("--sample_n",  type=int, default=None, help="隨機抽樣處理數量")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")),
                   key=os.path.getmtime)
    if args.latest and files:
        files = [files[-1]]
    elif args.sample_n and args.sample_n < len(files):
        files = random.sample(files, args.sample_n)

    outer = tqdm(files, desc="Encoding pipeline", unit="file")
    status = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for ply in outer:
        encode_one(ply, args.pf, args.K, args.num_bins, args.out_dir, status)
        outer.update(0)
    status.close()
    print("Encode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
