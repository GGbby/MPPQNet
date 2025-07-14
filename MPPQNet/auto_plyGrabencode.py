#!/usr/bin/env python3
# auto_plyGrabencode.py
# 1s/秒 擷取一次 ZED 點雲並壓縮
# 坐標用 MEASURE.XYZ，顏色用 VIEW.LEFT，避免 RGBA bits 解析錯誤

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
FPS           = 2.0
#———————#

os.makedirs(ZED_PLY_DIR,  exist_ok=True)
os.makedirs(BITSTREAM_DIR, exist_ok=True)

# 1. 只開一次相機
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode       = sl.DEPTH_MODE.ULTRA
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("[錯誤] ZED 開啟失敗"); sys.exit(1)
runtime = sl.RuntimeParameters()
mat     = sl.Mat()

last_ts = None

try:
    while True:
        t0 = time.time()
        # 2. grab
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(1/FPS); continue

        # 3. 取 XYZ
        zed.retrieve_measure(mat, sl.MEASURE.XYZ)
        xyz = mat.get_data()                   # shape=(H,W,3), float32

        # 4. 取顏色影像
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        img = mat.get_data()[..., :3]          # shape=(H,W,4)->(H,W,3), uint8

        # 5. flatten
        H,W,_ = xyz.shape
        pts = xyz.reshape(-1,3)
        cols = img.reshape(-1,3).astype(np.float32)/255.0

        # 6. 過濾無效點
        mask = (pts[:,2]>0) & np.isfinite(pts).all(axis=1)
        pts, cols = pts[mask], cols[mask]

        # 7. 寫 .ply
        ts = int(time.time())
        ply = os.path.join(ZED_PLY_DIR, f"{ts}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(ply, pcd)

        # 8. 只對新 ts 壓縮
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
                print(f"[訊息] {ts}.ply → {BITSTREAM_DIR}/{ts}_stream.bin")
                os.remove(ply); print(f"[訊息] 已刪除 {ts}.ply")
            except subprocess.CalledProcessError as e:
                print(f"[錯誤] 壓縮失敗: {e}")

        # 9. 控制頻率
        dt = time.time()-t0
        time.sleep(max(0, 1.0/FPS - dt))

except KeyboardInterrupt:
    pass
finally:
    zed.close()
    print("ZED 已關閉")
