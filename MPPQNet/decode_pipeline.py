import argparse
import os
import glob
import struct
import numpy as np
import open3d as o3d
from tqdm import tqdm

def decode_one(bin_path, out_dir, status_bar, use_gaussian):
    """
    讀取單一 .bin bitstream，依序解析 header、Stage1/Stage2 參數，
    並進行逐點還原與輸出 .ply。
    """
    base = os.path.splitext(os.path.basename(bin_path))[0].replace("_stream", "")
    # 1. 讀取 header
    status_bar.set_description(f"[{base}] 讀取 bitstream Header")
    status_bar.refresh()
    with open(bin_path, 'rb') as fb:
        # Header: num_bins, K, D, M (uint32 ×4)
        num_bins, K, D, M = struct.unpack('<IIII', fb.read(16))
        # 讀取最小值與最大值
        mn = np.frombuffer(fb.read(3*4), dtype=np.float32)
        mx = np.frombuffer(fb.read(3*4), dtype=np.float32)
        # Debug: 檢查 mn, mx 與差值
        print(f"[DEBUG] mn = {mn}, mx = {mx}, mx-mn = {mx - mn}")

        # 2. Stage1 thresholds & centers & μ & σ
        thr1_arr    = np.frombuffer(fb.read(D*(K-1)*4),   dtype=np.float32).reshape(D, K-1)
        cen1_arr    = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)
        mu1_arr     = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)
        sigma1_arr  = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)

        # 3. Stage2 thresholds & centers & μ & σ
        thr2_arr    = np.frombuffer(fb.read(D*(K-1)*4),   dtype=np.float32).reshape(D, K-1)
        cen2_arr    = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)
        mu2_arr     = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)
        sigma2_arr  = np.frombuffer(fb.read(D*K*4),       dtype=np.float32).reshape(D, K)

        # 4. Point-level labels1 & labels2 (uint8)
        labs1 = np.frombuffer(fb.read(M*D), dtype=np.uint8).reshape(M, D)
        labs2 = np.frombuffer(fb.read(M*D), dtype=np.uint8).reshape(M, D)
        # Debug: 檢查 Stage1 分佈
        print(f"[DEBUG] Stage1 x-dim thresholds: {thr1_arr[0]}")
        print(f"[DEBUG] Stage1 x-dim centers:    {cen1_arr[0]}")
        print(f"[DEBUG] Stage1 x-dim unique labels: {np.unique(labs1[:,0])}")
        # Debug: 檢查 Stage2 分佈
        print(f"[DEBUG] Stage2 x-dim thresholds: {thr2_arr[0]}")
        print(f"[DEBUG] Stage2 x-dim centers:    {cen2_arr[0]}")
        print(f"[DEBUG] Stage2 x-dim unique labels: {np.unique(labs2[:,0])}")

    # 5. Stage1 特徵還原
    status_bar.set_description(f"[{base}] Stage1 還原")
    status_bar.refresh()
    F1 = np.zeros((M, D), dtype=np.float32)
    for d in range(D):
        F1[:, d] = cen1_arr[d, labs1[:, d]]

    # 6. Stage2 殘差還原
    if use_gaussian:
        status_bar.set_description(f"[{base}] 殘差 Gaussian 還原")
        status_bar.refresh()
        R_hat = np.zeros((M, D), dtype=np.float32)
        for d in range(D):
            R_hat[:, d] = mu2_arr[d, labs2[:, d]]
    else:
        status_bar.set_description(f"[{base}] 殘差 centers 還原")
        status_bar.refresh()
        R_hat = np.zeros((M, D), dtype=np.float32)
        for d in range(D):
            R_hat[:, d] = cen2_arr[d, labs2[:, d]]

    # 7. 合併 Stage1 + Stage2
    status_bar.set_description(f"[{base}] 合併特徵")
    status_bar.refresh()
    F_hat = F1 + R_hat

    # 8. 拆解為座標與顏色
    status_bar.set_description(f"[{base}] 拆解座標與顏色")
    status_bar.refresh()
    coords = F_hat[:, :3]
    # 反歸一化回原始範圍
    coords = coords * (mx - mn) + mn

    # 9. 輸出重建 .ply
    status_bar.set_description(f"[{base}] 寫出 .ply")
    status_bar.refresh()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = np.clip(F_hat[:, 3:] / 255.0, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{base}_recon.ply"), pcd)


def main():
    parser = argparse.ArgumentParser(description="MPNN-based 3D 點雲解碼 (.bin bitstream)")
    parser.add_argument("--stream_dir", required=True, help="輸入 .bin bitstream 資料夾")
    parser.add_argument("--out_dir",    default="recon_ply", help="輸出重建 .ply 資料夾")
    parser.add_argument("--use_gaussian", action="store_true",
                        help="使用 Gaussian μ 還原殘差，否則使用 centers")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    streams = sorted(glob.glob(os.path.join(args.stream_dir, "*_stream.bin")))
    outer_bar  = tqdm(streams, desc="Decoding pipeline", unit="file", position=0)
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for bin_path in outer_bar:
        decode_one(bin_path, args.out_dir, status_bar, args.use_gaussian)
        outer_bar.update(0)
    status_bar.close()
    print("Decode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
