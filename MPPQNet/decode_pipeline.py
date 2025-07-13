# decode_pipeline.py
# 解碼 .bin bit-stream 為 .ply，使用 Product Quantization (PQ) 重建
# 支持前 N 筆帶字典，其後僅傳送 labels，並自動快取 fvecs 與 thresholds

import argparse
import os
import glob
import pickle

import numpy as np
import open3d as o3d
from tqdm import tqdm

# 全域緩存 PQ 字典：fvecs 與 thresholds
DICT_CACHE = {
    'fvecs':      None,  # numpy array shape (6, K)
    'thresholds': None,  # numpy array shape (6, K-1)
}

def decode_one(bin_path, out_dir, status_bar):
    """
    讀取並解碼單一 .bin 檔，輸出重建的 .ply
    支援先前快取字典 (fvecs, thresholds)，
    若 data 中帶有字典則更新快取
    """
    base = os.path.splitext(os.path.basename(bin_path))[0].replace("_stream", "")
    status_bar.set_description(f"[{base}] 讀取 bit-stream (.bin)")
    status_bar.refresh()

    # 1. 載入 bit-stream (pickle)
    with open(bin_path, 'rb') as f:
        data = pickle.load(f)

    # 2. 若 data 包含 fvecs 與 thresholds，則更新全域快取
    if 'fvecs' in data and 'thresholds' in data:
        DICT_CACHE['fvecs']      = data['fvecs']
        DICT_CACHE['thresholds'] = data['thresholds']

    fvecs     = DICT_CACHE['fvecs']      # (6, K)
    thr_arr   = DICT_CACHE['thresholds'] # (6, K-1)
    labels    = data['labels']           # (N, 6)

    # 3. 使用 labels 與 fvecs 重建特徵向量 F_hat (N,6)
    status_bar.set_description(f"[{base}] PQ 重建特徵向量")
    status_bar.refresh()
    N, D = labels.shape
    F_hat = np.zeros((N, D), dtype=np.float32)
    for d in range(D):
        F_hat[:, d] = fvecs[d, labels[:, d]]

    # 4. 拆解座標與顏色
    status_bar.set_description(f"[{base}] 拆解坐標與顏色")
    status_bar.refresh()
    coords = F_hat[:, :3]               # XYZ (歸一化範圍 0–1)
    colors = np.clip(F_hat[:, 3:] / 255.0, 0.0, 1.0)  # RGB 正規化至 [0,1]

    # 5. 輸出 .ply
    status_bar.set_description(f"[{base}] 寫出 .ply")
    status_bar.refresh()
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{base}_recon.ply"), pcd)


def main():
    parser = argparse.ArgumentParser(
        description="解碼 .bin PQ bit-stream 為 .ply，支援字典快取"
    )
    parser.add_argument(
        "--stream_dir", required=True,
        help="輸入 .bin bit-stream 資料夾，檔名格式 *_stream.bin"
    )
    parser.add_argument(
        "--out_dir", default="recon_ply_bin",
        help="輸出重建 .ply 資料夾"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bin_list = sorted(glob.glob(os.path.join(args.stream_dir, "*_stream.bin")))
    outer_bar = tqdm(bin_list, desc="Decoding pipeline", unit="file")
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)

    for bin_path in outer_bar:
        decode_one(bin_path, args.out_dir, status_bar)
        outer_bar.update(0)

    status_bar.close()
    print("Decode 完成 ▶", args.out_dir)


if __name__ == "__main__":
    main()
