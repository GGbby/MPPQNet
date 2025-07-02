# compute_metrics.py
import os
import glob
import argparse
import numpy as np
import open3d as o3d
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_d1_psnr(orig_pcd, recon_pcd):
    orig = np.asarray(orig_pcd.points)
    recon = np.asarray(recon_pcd.points)
    if orig.shape != recon.shape:
        raise ValueError(f"Point count mismatch: {orig.shape[0]} vs {recon.shape[0]}")
    mse = np.mean(np.sum((orig - recon)**2, axis=1))
    bbox = orig_pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    psnr = 20 * np.log10(diag / np.sqrt(mse + 1e-12))
    return mse, psnr

def compute_d2_psnr(orig_pcd, recon_pcd):
    oc = np.asarray(orig_pcd.colors)
    rc = np.asarray(recon_pcd.colors)
    if oc.size==0 or rc.size==0:
        return None, None
    if oc.max() <= 1.0:
        oc = (oc * 255).astype(np.float64)
        rc = (rc * 255).astype(np.float64)
    mse = np.mean(np.sum((oc - rc)**2, axis=1))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse + 1e-12))
    return mse, psnr

def main(orig_dir, recon_dir, bitstream_dir=None, out_csv=None):
    recon_files = glob.glob(os.path.join(recon_dir, '*_recon.ply'))
    if not recon_files:
        print(f"No reconstructed .ply files in {recon_dir}")
        return

    records = []
    for recon in tqdm(recon_files, desc='Processing', unit='ply'):
        name_method = os.path.splitext(os.path.basename(recon))[0]
        name, method = name_method.rsplit('_', 1)
        orig_path = os.path.join(orig_dir, f"{name}.ply")
        if not os.path.exists(orig_path):
            continue
        orig_pcd = o3d.io.read_point_cloud(orig_path)
        recon_pcd = o3d.io.read_point_cloud(recon)
        mse1, ps1 = compute_d1_psnr(orig_pcd, recon_pcd)
        mse2, ps2 = compute_d2_psnr(orig_pcd, recon_pcd)
        num_pts = np.asarray(orig_pcd.points).shape[0]
        bpp = None
        if bitstream_dir:
            bs = os.path.join(bitstream_dir, f"{name}_stream.npz")
            if os.path.exists(bs):
                size_bits = os.path.getsize(bs) * 8
                bpp = size_bits / num_pts
        records.append((bpp, ps1, ps2))

    # 如果有 bitstream_dir，繪製 RD 曲線
    if bitstream_dir:
        rec = [r for r in records if r[0] is not None]
        rec.sort(key=lambda x: x[0])
        bpps, ps1s, ps2s = zip(*rec)
        plt.figure()
        plt.plot(bpps, ps1s, marker='o', label='D1 PSNR')
        plt.plot(bpps, ps2s, marker='s', label='D2 PSNR')
        plt.xlabel('Bits per point')
        plt.ylabel('PSNR (dB)')
        plt.title('Rate-Distortion Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 輸出 CSV
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bpp','psnr_d1','psnr_d2'])
            for bpp, ps1, ps2 in records:
                writer.writerow([bpp or '', ps1, ps2])
        print(f"Saved CSV ▶ {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics & optional RD plot')
    parser.add_argument('--orig_dir',      required=True, help='原始 .ply 資料夾')
    parser.add_argument('--recon_dir',     required=True, help='重建 .ply 資料夾')
    parser.add_argument('--bitstream_dir', help='用於計算 bpp 的 .npz bit-stream 資料夾')
    parser.add_argument('--csv',           help='輸出 CSV 路徑')
    args = parser.parse_args()
    main(args.orig_dir, args.recon_dir, args.bitstream_dir, args.csv)
