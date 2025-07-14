#!/usr/bin/env python3
# auto_plyGrabencode.py
# 每秒擷取一次 ZED 點雲 (.ply)，呼叫 encode_pipeline 壓縮 (.bin)，刪除暫存 .ply
# 以兩個不同 sl.Mat 物件分別做 XYZ 與 RGB 讀取，避免 segmentation fault

import time, os, sys, subprocess
import numpy as np
import open3d as o3d
import pyzed.sl as sl

# —— 參數 ——#
ZED_PLY_DIR   = 'zed_plys'
BITSTREAM_DIR = 'bitstreams_bin'
PF_WEIGHT     = './pth/b128/mppn_pf_k128b128.pth'
K             = 128
NUM_BINS      = 128
FPS           = 1.0
#———————#

os.makedirs(ZED_PLY_DIR,  exist_ok=True)
os.makedirs(BITSTREAM_DIR, exist_ok=True)

# 1. 只開一次 ZED 相機
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode        = sl.DEPTH_MODE.ULTRA
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("[錯誤] ZED 開啟失敗"); sys.exit(1)

runtime = sl.RuntimeParameters()
mat_xyz = sl.Mat()    # 專門拿深度
mat_img = sl.Mat()    # 專門拿影像

last_ts = None

try:
    while True:
        t0 = time.time()
        # 2. Grab
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(1/FPS)
            continue

        # 3. 讀 XYZ
        zl = zed.retrieve_measure(mat_xyz, sl.MEASURE.XYZ)
        if zl != sl.ERROR_CODE.SUCCESS:
            print("[錯誤] 讀取 XYZ 失敗")
            time.sleep(1/FPS)
            continue
        xyz = mat_xyz.get_data()             # (H, W, 3) float32

        # 4. 讀 RGB
        zr = zed.retrieve_image(mat_img, sl.VIEW.LEFT)
        if zr != sl.ERROR_CODE.SUCCESS:
            print("[錯誤] 讀取 LEFT image 失敗")
            time.sleep(1/FPS)
            continue
        img = mat_img.get_data()[..., :3]    # (H, W, 3) uint8

        # 5. Flatten
        H, W, _ = xyz.shape
        pts  = xyz.reshape(-1, 3)
        cols = img.reshape(-1,3).astype(np.float32) / 255.0

        # 6. 過濾 z<=0 or NaN
        mask = (pts[:,2] > 0) & np.isfinite(pts).all(axis=1)
        pts, cols = pts[mask], cols[mask]

        # 7. 寫出暫存 .ply
        ts = int(time.time())
        ply_path = os.path.join(ZED_PLY_DIR, f"{ts}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        if not o3d.io.write_point_cloud(ply_path, pcd):
            print(f"[錯誤] 無法寫出 {ply_path}")
            time.sleep(1/FPS)
            continue

        # 8. 只處理新 ts
        if ts != last_ts:
            last_ts = ts
            cmd = [
                sys.executable, 'encode_pipeline.py',
                '--input_dir', ZED_PLY_DIR,
                '--pf',         PF_WEIGHT,
                '--latest',
                '--K',          str(K),
                '--num_bins',   str(NUM_BINS),
                '--out_dir',    BITSTREAM_DIR
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"[訊息] 已壓縮 {ts}.ply → {BITSTREAM_DIR}/{ts}_stream.bin")
                os.remove(ply_path)
                print(f"[訊息] 已刪除暫存 .ply：{ts}.ply")
            except subprocess.CalledProcessError as e:
                print(f"[錯誤] 壓縮失敗: {e}")

        # 9. 控制頻率
        elapsed = time.time() - t0
        time.sleep(max(0, 1.0/FPS - elapsed))

except KeyboardInterrupt:
    pass
finally:
    zed.close()
    print("ZED 已關閉，程式結束")
