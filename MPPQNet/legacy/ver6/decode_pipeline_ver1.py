import argparse
import os
import glob
import struct
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

def decode_one(bin_path, out_dir, status_bar, use_gaussian):
    base = os.path.splitext(os.path.basename(bin_path))[0].replace("_stream", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 讀取 header
    status_bar.set_description(f"[{base}] 讀取 bitstream Header")
    status_bar.refresh()
    with open(bin_path, 'rb') as fb:
        num_bins, K, D, M = struct.unpack('<IIII', fb.read(16))
        mn = np.frombuffer(fb.read(3*4), dtype=np.float32)
        mx = np.frombuffer(fb.read(3*4), dtype=np.float32)
        thr1_arr = np.frombuffer(fb.read(D*(K-1)*4), dtype=np.float32).reshape(D, K-1)
        cen1_arr = np.frombuffer(fb.read(D*K*4),   dtype=np.float32).reshape(D, K)
        mu1_arr  = np.frombuffer(fb.read(D*K*4),   dtype=np.float32).reshape(D, K)
        sigma1_arr = np.frombuffer(fb.read(D*K*4), dtype=np.float32).reshape(D, K)
        thr2_arr = np.frombuffer(fb.read(D*(K-1)*4), dtype=np.float32).reshape(D, K-1)
        cen2_arr = np.frombuffer(fb.read(D*K*4),   dtype=np.float32).reshape(D, K)
        mu2_arr  = np.frombuffer(fb.read(D*K*4),   dtype=np.float32).reshape(D, K)
        sigma2_arr = np.frombuffer(fb.read(D*K*4), dtype=np.float32).reshape(D, K)
        labs1 = np.frombuffer(fb.read(M*D), dtype=np.uint8).reshape(M, D)
        labs2 = np.frombuffer(fb.read(M*D), dtype=np.uint8).reshape(M, D)

    # 2. 轉為 torch Tensor 並搬到 GPU
    thr1 = torch.from_numpy(thr1_arr).to(device)
    cen1 = torch.from_numpy(cen1_arr).to(device)
    labs1_t = torch.from_numpy(labs1).long().to(device)
    thr2 = torch.from_numpy(thr2_arr).to(device)
    cen2 = torch.from_numpy(cen2_arr).to(device)
    labs2_t = torch.from_numpy(labs2).long().to(device)
    mn_t = torch.from_numpy(mn).view(1,3).to(device)
    mx_t = torch.from_numpy(mx).view(1,3).to(device)

    # 3. Stage1 還原 on GPU
    status_bar.set_description(f"[{base}] Stage1 還原 (GPU)")
    status_bar.refresh()
    F1_t = torch.zeros((M, D), dtype=torch.float32, device=device)
    for d in range(D):
        F1_t[:, d] = cen1[d, labs1_t[:, d]]

    # 4. Stage2 殘差還原 on GPU
    if use_gaussian:
        status_bar.set_description(f"[{base}] 殘差 Gaussian 還原 (GPU)")
    else:
        status_bar.set_description(f"[{base}] 殘差 centers 還原 (GPU)")
    status_bar.refresh()
    R_hat_t = torch.zeros((M, D), dtype=torch.float32, device=device)
    for d in range(D):
        if use_gaussian:
            R_hat_t[:, d] = mu2_arr[d, labs2[:, d]] if isinstance(mu2_arr, torch.Tensor) else torch.from_numpy(mu2_arr)[d, labs2_t[:, d]].to(device)
            # but better convert mu2_arr to torch above
            R_hat_t[:, d] = torch.from_numpy(mu2_arr).to(device)[d, labs2_t[:, d]]
        else:
            R_hat_t[:, d] = cen2[d, labs2_t[:, d]]

    # 5. 合併特徵
    status_bar.set_description(f"[{base}] 合併特徵 (GPU)")
    status_bar.refresh()
    F_hat_t = F1_t + R_hat_t

    # 6. 拆解座標與顏色
    status_bar.set_description(f"[{base}] 拆解座標與顏色 (GPU)")
    status_bar.refresh()
    coords_t = F_hat_t[:, :3] * (mx_t - mn_t) + mn_t
    colors_t = torch.clamp(F_hat_t[:, 3:] / 255.0, 0.0, 1.0)

    # 7. 回 CPU 並輸出 .ply
    status_bar.set_description(f"[{base}] 寫出 .ply")
    status_bar.refresh()
    coords = coords_t.cpu().numpy()
    colors = colors_t.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{base}_recon.ply"), pcd)


def main():
    parser = argparse.ArgumentParser(description="MPNN-based 3D 點雲解碼 (.bin bitstream)")
    parser.add_argument("--stream_dir", required=True, help="輸入 .bin bitstream 資料夾")
    parser.add_argument("--out_dir", default="recon_ply", help="輸出重建 .ply 資料夾")
    parser.add_argument("--use_gaussian", action="store_true", help="使用 Gaussian μ 還原殘差")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    streams = sorted(glob.glob(os.path.join(args.stream_dir, "*_stream.bin")))
    outer_bar = tqdm(streams, desc="Decoding pipeline", unit="file", position=0)
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for bin_path in outer_bar:
        decode_one(bin_path, args.out_dir, status_bar, args.use_gaussian)
        outer_bar.update(0)
    status_bar.close()
    print("Decode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
