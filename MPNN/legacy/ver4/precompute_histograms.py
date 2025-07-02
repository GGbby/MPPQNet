# precompute_histograms.py

import os
import glob
import argparse
import numpy as np
import open3d as o3d
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from tqdm import tqdm

def compute_and_save_hist(ply_path, out_path, num_bins):
    """
    對單一 .ply 檔計算 6 條直方圖，並儲存為一個 .npz 檔
    參數:
      ply_path: 原始 .ply 檔案路徑
      out_path: 輸出的 .npz 檔案路徑，例如 train/sceneA_hist.npz
      num_bins: 直方圖的 bin 數量
    回傳: 無，將直方圖存入 out_path
    """
    # 讀取點雲並取得 XYZ 與 RGB
    pcd    = o3d.io.read_point_cloud(ply_path)
    pts    = np.asarray(pcd.points)    # 形狀 (M,3)
    colors = np.asarray(pcd.colors)    # 形狀 (M,3)、值域 [0,1]

    # 建立特徵矩陣 F (M,6)
    F = build_feature_matrix(pts, colors)

    # 計算每個維度的直方圖 (共 6 維)
    hists = compute_histogram_per_column(F, num_bins)

    # 儲存直方圖到 .npz，key 為 'hists'，shape = (6, num_bins)
    np.savez(out_path, hists=np.stack(hists, axis=0))


def main(source_dir, out_dir, num_bins):
    """
    遞迴讀取 source_dir 底下所有子資料夾的 .ply 檔，
    計算並儲存直方圖到 out_dir。
    參數:
      source_dir: 包含 .ply 的根資料夾（遞迴查找）
      out_dir:    儲存 .npz 檔的目錄
      num_bins:   直方圖的 bin 數量
    """
    # 確保輸出目錄存在
    os.makedirs(out_dir, exist_ok=True)

    # 遞迴收集所有 .ply 檔路徑
    all_plys = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".ply"):
                all_plys.append(os.path.join(root, f))

    if not all_plys:
        print("⚠️ 未在指定目錄找到任何 .ply 檔，請檢查 source_dir 是否正確。")
        return

    # 使用 tqdm 顯示處理進度
    for ply_path in tqdm(all_plys, desc="預計算 6 維直方圖", unit="file"):
        base     = os.path.splitext(os.path.basename(ply_path))[0]
        out_name = f"{base}_hist.npz"
        out_path = os.path.join(out_dir, out_name)
        compute_and_save_hist(ply_path, out_path, num_bins)

    print(f"✅ 已完成 {len(all_plys)} 個 .ply 檔的 6 維直方圖預計算，結果儲存在：{out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預計算 .ply 的 6 維直方圖並儲存為 .npz 檔")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="包含 .ply 檔的根資料夾，程式會遞迴搜尋所有子目錄。")
    parser.add_argument("--out_dir",    type=str, required=True,
                        help="預計算後直方圖 .npz 檔要儲存的資料夾。")
    parser.add_argument("--num_bins",   type=int, default=64,
                        help="直方圖 bin 數量，預設 64。")
    args = parser.parse_args()

    main(args.source_dir, args.out_dir, args.num_bins)
