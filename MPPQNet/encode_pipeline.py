# encode_pipeline.py
# 在 GPU (CUDA) 上使用 PFNetwork 進行 6 維特徵乘積量化
# 自動過濾 NaN/inf 點並跳過空點雲
# 使用 zlib 壓縮 pickle 流以減少 bit-stream 大小

import argparse
import os
import glob
import random
import pickle
import zlib                        # 用於壓縮 pickle 流

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from feature_extraction import build_feature_matrix
from mpnn_qnn import run_mppn


def encode_one(ply_path, pf_weight, K, num_bins, out_dir, status_bar):
    base = os.path.splitext(os.path.basename(ply_path))[0]

    # 1. 讀取點雲並過濾 NaN/inf
    status_bar.set_description(f"[{base}] 讀取與過濾點雲")
    status_bar.refresh()
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        print(f"[警告] {base}.ply 無點，跳過")
        return
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    cols = np.asarray(pcd.colors, dtype=np.float32)[finite_mask]
    if pts.shape[0] == 0:
        print(f"[警告] {base}.ply 全為 NaN/inf，跳過")
        return

    # 2. 建立 6D 特徵矩陣 (XYZ 歸一化至 [0,1] + RGB 0–255)
    status_bar.set_description(f"[{base}] 建立特徵矩陣")
    status_bar.refresh()
    min_c, max_c = pts.min(axis=0), pts.max(axis=0)
    coords_norm = (pts - min_c) / (max_c - min_c + 1e-8)
    colors      = (cols * 255.0).astype(np.float32)
    F = build_feature_matrix(coords_norm, colors)  # shape: (M,6)

    # 3. PFNetwork 推理 (GPU histogram + MPNN)
    status_bar.set_description(f"[{base}] PF 推理 (K={K}, bins={num_bins})")
    status_bar.refresh()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_t = torch.from_numpy(F).to(device)  # (M,6)

    fvecs_list = []
    thr_list   = []
    for d in range(F_t.shape[1]):
        # GPU 上計算直方圖
        col = F_t[:, d]
        h = torch.histc(col, bins=num_bins,
                        min=col.min().item(),
                        max=col.max().item())
        h = h / h.sum()
        H = h.cpu().numpy().astype(np.float32)

        # 執行一次 forward，載入預訓練權重
        _, f_vec, thr = run_mppn(
            H, K=K, epochs=1, device=device, pf_weight=pf_weight
        )
        fvecs_list.append(f_vec)
        thr_list.append(thr)

    # 把 list 轉成陣列，fvecs: (6,K), thr: (6,K-1)
    fvecs_arr = np.stack(fvecs_list, axis=0)
    thr_arr   = np.stack(thr_list,   axis=0)

    # 4. Bucketize + 重建 (全在 GPU)
    status_bar.set_description(f"[{base}] 量化與重建")
    status_bar.refresh()
    labels_t = torch.zeros_like(F_t, dtype=torch.long)
    thr_t_list = [torch.from_numpy(t).to(device) for t in thr_arr]
    fvecs_t    = torch.from_numpy(fvecs_arr).to(device)

    for d in range(F_t.shape[1]):
        col = F_t[:, d].contiguous()         # 先強制成 contiguous
        idx = torch.bucketize(col, thr_t_list[d])
        labels_t[:, d] = idx

    labels = labels_t.cpu().numpy().astype(np.uint16)  # (M,6) 存成 uint16

    # 5. 序列化 payload
    payload = {
        'shape':      F.shape,       # 原始點雲大小 (M,6)
        'fvecs':      fvecs_arr,     # Product Quantization 字典
        'thresholds': thr_arr,       # 分界閾值
        'labels':     labels,        # 量化後索引
    }

    # 6. pickle + zlib 壓縮，寫成 .bin
    status_bar.set_description(f"[{base}] 壓縮並輸出 .bin")
    status_bar.refresh()
    raw = pickle.dumps(payload)           # dump 成 bytes
    comp = zlib.compress(raw, level=9)    # zlib 最大壓縮等級
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_stream.bin")
    with open(out_path, 'wb') as f:
        f.write(comp)                     # 寫入壓縮後 bytes


def main():
    parser = argparse.ArgumentParser(
        description="GPU PFNetwork 量化：.ply -> .bin (zlib 壓縮)"
    )
    parser.add_argument("--input_dir",  required=True, help="原始 .ply 資料夾")
    parser.add_argument("--pf",         required=True, help="PFNetwork 權重檔 (.pth)")
    parser.add_argument("--K",   type=int, default=6,    help="PFNetwork 類別數 K")
    parser.add_argument("--num_bins",    type=int, default=64,  help="直方圖 bins 數")
    parser.add_argument("--out_dir",     default="bitstreams_bin", help="輸出 .bin 資料夾")
    parser.add_argument("--sample_n",    type=int, default=None,  help="隨機抽樣檔案數")
    parser.add_argument("--latest",      action="store_true",      help="僅處理最新一個 .ply")
    args = parser.parse_args()

    # 只處理最新一個或隨機 n 個 ply
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
