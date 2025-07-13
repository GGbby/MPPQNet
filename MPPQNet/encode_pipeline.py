# encode_pipeline.py
# GPU 上使用 PFNetwork 進行 6 維特徵乘積量化，並自動過濾 ZED 產生的 NaN 點雲

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

    # 1. 讀取點雲並建 6D 特徵矩陣 (XYZ 歸一化 + RGB)
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points, dtype=np.float32)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    coords_norm = (coords - min_c) / (max_c - min_c + 1e-8)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # shape: (N,6)

    # 2. PFNetwork 推理 (GPU histogram + MPNN)
    status_bar.set_description(f"[{base}] PF 推理 (K={K}, bins={num_bins})")
    status_bar.refresh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_t = torch.from_numpy(F).to(device)  # (N,6)

    # 2.1 過濾 NaN 行
    valid_mask = ~torch.isnan(F_t).any(dim=1)
    if valid_mask.sum() == 0:
        raise RuntimeError(f"[{base}] 全部點皆為 NaN，無法壓縮")
    F_valid = F_t[valid_mask]  # (M_valid,6)

    fvecs_list = []
    thr_list   = []
    for d in range(F_valid.shape[1]):
        col = F_valid[:, d]
        # GPU 上計算直方圖
        h = torch.histc(col, bins=num_bins,
                        min=col.min().item(),
                        max=col.max().item())
        h = h / h.sum()
        H = h.cpu().numpy().astype(np.float32)

        # MPPN forward with pretrained PFNetwork
        _, f_vec, thr = run_mppn(
            H, K=K, epochs=1, device=device, pf_weight=pf_weight
        )
        fvecs_list.append(f_vec)
        thr_list.append(thr)

    fvecs_arr = np.stack(fvecs_list, axis=0)  # (6, K)
    thr_arr   = np.stack(thr_list,   axis=0)  # (6, K-1)

    # 3. 用 bucketize 做量化標籤並重建特徵
    status_bar.set_description(f"[{base}] 量化編碼")
    status_bar.refresh()
    labels_t = torch.zeros_like(F_t, dtype=torch.long)
    recon_t  = torch.zeros_like(F_t)
    thr_t_list = [torch.from_numpy(t).to(device) for t in thr_arr]
    fvecs_t    = torch.from_numpy(fvecs_arr).to(device)

    # 3.1 只對 valid 行做 bucketize
    for d in range(F_valid.shape[1]):
        thr_t = thr_t_list[d]
        idx   = torch.bucketize(F_valid[:, d], thr_t)  # (M_valid,)
        labels_t[valid_mask, d] = idx
        recon_t[valid_mask, d]  = fvecs_t[d][idx]

    labels      = labels_t.cpu().numpy()    # (N,6)
    recon_feats = recon_t.cpu().numpy()     # (N,6)

    # 4. 序列化輸出 .bin
    status_bar.set_description(f"[{base}] 寫出 bit-stream (.bin)")
    status_bar.refresh()
    payload = {
        'shape':       F.shape,
        'fvecs':       fvecs_arr,
        'thresholds':  thr_arr,
        'labels':      labels,
        'recon_feats': recon_feats
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(
        description="Encode .ply -> .bin using GPU PFNetwork with NaN-filtering"
    )
    parser.add_argument("--input_dir",  required=True, help="來源 .ply 資料夾")
    parser.add_argument("--pf",         required=True, help="PFNetwork 權重檔 (.pth)")
    parser.add_argument("--K",   type=int, default=6,    help="PFNetwork 類別數 K")
    parser.add_argument("--num_bins",    type=int, default=64,   help="直方圖 bins 數")
    parser.add_argument("--out_dir",     default="bitstreams_bin", help="輸出 .bin 資料夾")
    parser.add_argument("--sample_n",    type=int, default=None, help="隨機抽樣處理數量")
    parser.add_argument("--latest",      action="store_true",     help="只處理最後修改的 .ply")
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
        encode_one(
            ply, args.pf, args.K, args.num_bins,
            args.out_dir, status
        )
        outer.update(0)
    status.close()
    print("Encode 完成 ▶", args.out_dir)


if __name__ == '__main__':
    main()
