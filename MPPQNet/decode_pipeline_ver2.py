# decode_pipeline_v2.py
# 針對 encode_pipeline.py (v2) 輸出的 .npz bit-stream 進行解碼，
# 支持使用 PFNetwork soft-VQ 重構 (pf_codewords) 與可選 Gaussian 殘差還原

import argparse
import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm


def decode_one(npz_path, out_dir, status_bar, use_gaussian):
    base = os.path.splitext(os.path.basename(npz_path))[0].replace("_stream", "")
    status_bar.set_description(f"[{base}] 讀取 bit-stream (.npz)")
    status_bar.refresh()
    # 載入所有編碼資料
    data = np.load(npz_path)

    # 1. Stage1 coarse 還原: 使用 soft-VQ 重構結果 pf_codewords
    if 'pf_codewords' in data:
        F1 = data['pf_codewords']  # 已包含 soft assignment 重構特徵
    else:
        # 備援: 若無 soft-VQ，則由 pt_labels & pt_centers 還原
        labels_pt = data['pt_labels']
        centers_pt = data['pt_centers']
        F1 = centers_pt[labels_pt]

    # 2. Stage2 殘差還原
    if use_gaussian:
        status_bar.set_description(f"[{base}] 殘差 Gaussian 還原")
        status_bar.refresh()
        mu_res = data['gauss_mu_res']        # Gaussian μ
        R_hat = mu_res[data['res_labels']]
    else:
        status_bar.set_description(f"[{base}] 殘差中心還原")
        status_bar.refresh()
        centers_res = data['res_centers']
        R_hat = centers_res[data['res_labels']]

    # 3. 合併特徵
    status_bar.set_description(f"[{base}] 合併特徵")
    status_bar.refresh()
    F_hat = F1 + R_hat

    # 4. 拆解坐標與顏色
    status_bar.set_description(f"[{base}] 拆解坐標與顏色")
    status_bar.refresh()
    coords = F_hat[:, :3]             # 已為 [0,1] 或原尺度 (依 encode)
    colors = np.clip(F_hat[:, 3:] / 255.0, 0.0, 1.0)

    # 5. 輸出 .ply
    status_bar.set_description(f"[{base}] 寫出 .ply")
    status_bar.refresh()
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = os.path.join(out_dir, f"{base}_recon.ply")
    o3d.io.write_point_cloud(out_path, pcd)


def main():
    parser = argparse.ArgumentParser(
        description="解碼 .npz bit-stream 為 .ply，支持 soft-VQ 與 Gaussian 殘差"
    )
    parser.add_argument(
        "--stream_dir", required=True,
        help="輸入 .npz bit-stream 資料夾"
    )
    parser.add_argument(
        "--out_dir", default="recon_ply_v2",
        help="輸出重建 .ply 資料夾"
    )
    parser.add_argument(
        "--use_gaussian", action="store_true",
        help="使用 Gaussian μ 還原殘差，否則使用 cluster center"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npz_list = sorted(glob.glob(os.path.join(args.stream_dir, "*_stream.npz")))
    outer_bar = tqdm(npz_list, desc="Decoding pipeline", unit="file")
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for npz_path in outer_bar:
        decode_one(npz_path, args.out_dir, status_bar, args.use_gaussian)
        outer_bar.update(0)
    status_bar.close()
    print("Decode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
