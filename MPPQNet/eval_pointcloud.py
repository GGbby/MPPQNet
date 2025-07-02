import os
import glob
import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

# 計算 MSE 與 RMSE
def compute_rmse(orig_pts, recon_pts):
    """假設一對一對應，計算 MSE 與 RMSE"""
    diff = orig_pts - recon_pts
    mse = np.mean(np.sum(diff**2, axis=1))
    rmse = np.sqrt(mse)
    return mse, rmse

# 計算 Chamfer Distance
def compute_chamfer(orig_pts, recon_pts):
    """
    雙向最近鄰歐氏距離平均 (Chamfer Distance)，
    回傳 d1 = avg_{o∈orig} min_{r∈recon} ||o-r||,
         d2 = avg_{r∈recon} min_{o∈orig} ||r-o||,
         cd = d1 + d2
    """
    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(recon_pts)
    tree_r = o3d.geometry.KDTreeFlann(pcd_r)
    d1_list = [np.sqrt(tree_r.search_knn_vector_3d(p,1)[2][0]) for p in orig_pts]

    pcd_o = o3d.geometry.PointCloud()
    pcd_o.points = o3d.utility.Vector3dVector(orig_pts)
    tree_o = o3d.geometry.KDTreeFlann(pcd_o)
    d2_list = [np.sqrt(tree_o.search_knn_vector_3d(p,1)[2][0]) for p in recon_pts]

    d1_avg = float(np.mean(d1_list))
    d2_avg = float(np.mean(d2_list))
    return d1_avg, d2_avg, d1_avg + d2_avg

# 轉換位元數為易讀格式
def human_size(nbytes):
    for unit in ['B','KB','MB','GB']:
        if nbytes < 1024.0:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}TB"

# 主程式入口
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_dir",   required=True, help="原始 .ply 資料夾")
    parser.add_argument("--stream_dir", required=True, help="壓縮後 .bin 資料夾")
    parser.add_argument("--recon_dir",  required=True, help="重建後 .ply 資料夾")
    parser.add_argument("--out_csv",    default="evaluation_results.csv", help="輸出 CSV 檔名")
    args = parser.parse_args()

    orig_list = sorted(glob.glob(os.path.join(args.orig_dir, "*.ply")))
    records = []

    for orig_path in tqdm(orig_list, desc="Evaluation", unit="file"):
        base = os.path.splitext(os.path.basename(orig_path))[0]
        stream_path = os.path.join(args.stream_dir, base + "_stream.bin")
        recon_path = os.path.join(args.recon_dir, base + "_recon.ply")
        if not (os.path.exists(stream_path) and os.path.exists(recon_path)):
            records.append({"file": base, "error": "missing files"})
            continue

        size_o = os.path.getsize(orig_path)
        size_s = os.path.getsize(stream_path)
        cratio = size_o / size_s

        pts_o = np.asarray(o3d.io.read_point_cloud(orig_path).points)
        pts_r = np.asarray(o3d.io.read_point_cloud(recon_path).points)

        mse, rmse = compute_rmse(pts_o, pts_r)
        d1, d2, cd = compute_chamfer(pts_o, pts_r)

        num_points     = pts_o.shape[0]
        num_out_points = pts_r.shape[0]
        stream_bits = size_s * 8
        bpip = stream_bits / num_points
        bpop = stream_bits / num_out_points
        MAX_G = 2**10 - 1
        psnr_d1 = 10 * np.log10((MAX_G ** 2) / mse)

        records.append({
            "file": base,
            "orig_size": human_size(size_o),
            "stream_size": human_size(size_s),
            "cratio": f"{cratio:.2f}×",
            "num_points": num_points,
            "num_out_points": num_out_points,
            "bpip": f"{bpip:.6f}",
            "bpop": f"{bpop:.6f}",
            "mse": f"{mse:.6f}",
            "rmse": f"{rmse:.6f}",
            "psnr_d1": f"{psnr_d1:.2f}",
            "d1": f"{d1:.6f}",
            "d2": f"{d2:.6f}",
            "chamfer": f"{cd:.6f}"
        })

    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
