# decode_pipeline_v2_bin.py
# 解碼 .bin binary bit-stream 為 .ply，支持 soft-VQ 與 Gaussian 殘差
# 使用 Python pickle 作為二進位序列化格式

import argparse
import os
import glob
import pickle
import numpy as np
import open3d as o3d
from tqdm import tqdm

# 全域緩存字典
DICT_CACHE = {}

def decode_one(bin_path, out_dir, status_bar, use_gaussian):
    base = os.path.splitext(os.path.basename(bin_path))[0].replace("_stream", "")
    status_bar.set_description(f"[{base}] 讀取 bit-stream (.bin)")
    status_bar.refresh()
    # 讀 pickle
    with open(bin_path, 'rb') as f:
        data = pickle.load(f)

    # 1. Stage1 還原 (coarse)
    status_bar.set_description(f"[{base}] Stage1 還原 (coarse)")
    status_bar.refresh()
    if 'pf_codewords' in data:
        F1 = data['pf_codewords']
    else:
        F1 = data['pt_centers'][data['pt_labels']]

    # 2. Stage2 殘差還原
    if use_gaussian:
        status_bar.set_description(f"[{base}] Stage2 殘差 Gaussian 還原")
        status_bar.refresh()
        mu_res = data.get('gauss_mu_res', DICT_CACHE.get('gauss_mu_res'))
        R_hat = mu_res[data['res_labels']]
    else:
        status_bar.set_description(f"[{base}] Stage2 殘差中心還原")
        status_bar.refresh()
        centers_res = data.get('res_centers', DICT_CACHE.get('res_centers'))
        R_hat = centers_res[data['res_labels']]

    # 緩存字典
    if 'gauss_mu_res' in data and 'res_centers' in data:
        DICT_CACHE['gauss_mu_res'] = data['gauss_mu_res']
        DICT_CACHE['res_centers'] = data['res_centers']

    # 3. 合併特徵
    status_bar.set_description(f"[{base}] 合併特徵")
    status_bar.refresh()
    F_hat = F1 + R_hat

    # 4. 拆解坐標與顏色
    status_bar.set_description(f"[{base}] 拆解坐標與顏色")
    status_bar.refresh()
    coords = F_hat[:, :3]
    colors = np.clip(F_hat[:, 3:] / 255.0, 0.0, 1.0)

    # 5. 寫出 .ply
    status_bar.set_description(f"[{base}] 寫出 .ply")
    status_bar.refresh()
    os.makedirs(out_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, f"{base}_recon.ply"), pcd)


def main():
    parser = argparse.ArgumentParser(
        description="解碼 .bin bit-stream 為 .ply，支持 soft-VQ 與 Gaussian 殘差"
    )
    parser.add_argument("--stream_dir", required=True, help="輸入 .bin bit-stream 資料夾")
    parser.add_argument("--out_dir", default="recon_ply_bin", help="輸出 .ply 資料夾")
    parser.add_argument("--use_gaussian", action="store_true", help="使用 Gaussian μ 還原殘差")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bin_list = sorted(glob.glob(os.path.join(args.stream_dir, "*_stream.bin")))
    outer_bar = tqdm(bin_list, desc="Decoding pipeline", unit="file")
    status_bar = tqdm(total=0, bar_format="{desc}", position=1, leave=True)
    for bin_path in outer_bar:
        decode_one(bin_path, args.out_dir, status_bar, args.use_gaussian)
        outer_bar.update(0)
    status_bar.close()
    print("Decode 完成 ▶", args.out_dir)

if __name__ == "__main__":
    main()
