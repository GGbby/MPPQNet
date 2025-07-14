# auto_plyGrabencode.py
# 持續擷取 ZED 點雲，並以高頻率送入 encode_pipeline.py 壓縮
# 改為：只開一次 ZED Camera，再在迴圈中 grab

import time
import os
import subprocess
import sys

# ZED Python SDK
import pyzed.sl as sl
import numpy as np
import open3d as o3d

#—— 參數設定 ——#
ZED_PLY_DIR   = 'zed_plys'        # 存放暫存 .ply 資料夾
BITSTREAM_DIR = 'bitstreams_bin'   # 輸出 .bin 壓縮檔資料夾
PF_WEIGHT     = './pth/b128/mppn_pf_k128b128.pth'  # PFNetwork 權重檔
K             = 128                # PFNetwork 類別數
NUM_BINS      = 128                # histogram bins
FPS           = 5.0                # 擷取與壓縮頻率 (Hz)，可自由調整
#—————————————#

# 建資料夾
os.makedirs(ZED_PLY_DIR,  exist_ok=True)
os.makedirs(BITSTREAM_DIR, exist_ok=True)

# 1. 只開一次 ZED 相機
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.camera_resolution = sl.RESOLUTION.HD720
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"[錯誤] ZED 開啟失敗：{err}")
    sys.exit(1)

runtime = sl.RuntimeParameters()
last_ts = None

try:
    while True:
        start = time.time()
        # 2. 擷取影像與深度
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("[錯誤] ZED 擷取失敗")
            time.sleep(1.0 / FPS)
            continue

        # 3. 同步取得 XYZ + RGBA bits
        mat = sl.Mat()
        zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
        xyzrgba = mat.get_data()           # float32, shape=(H,W,4)

        # 4. 展平並拆出 XYZ 與 RGB
        H, W, _ = xyzrgba.shape
        # XYZ
        pts = xyzrgba[..., :3].reshape(-1, 3)   # (H*W,3)
        # RGBA bits 存在第 4 channel
        rgba_flat = xyzrgba[..., 3].reshape(-1).astype(np.uint32)  # (H*W,)
        rgb_int   = rgba_flat & 0x00FFFFFF                         # 取低 24 bit
        # normalize 到 [0,1]
        cols      = (rgb_int.astype(np.float32) / float(2**24 - 1)).reshape(-1,1)
        cols      = np.repeat(cols, 3, axis=1)                     # (H*W,3)

        # 5. 過濾掉無效或 z<=0 點
        mask = (pts[:,2] > 0) & np.isfinite(pts).all(axis=1)
        pts, cols = pts[mask], cols[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
        o3d.io.write_point_cloud(ply_path, pcd)

        # 6. 只在新檔才做壓縮
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
                print(f"[訊息] 已壓縮 {ts}.ply → bitstreams_bin/{ts}_stream.bin")
                # 刪除暫存 ply
                os.remove(ply_path)
                print(f"[訊息] 已刪除暫存 ply：{ts}.ply")
            except subprocess.CalledProcessError as e:
                print(f"[錯誤] 壓縮失敗: {e}")

        # 7. 控制頻率
        elapsed = time.time() - start
        to_sleep = max(0, (1.0 / FPS) - elapsed)
        time.sleep(to_sleep)

finally:
    # 8. 退出：關閉相機
    zed.close()
    print("ZED 已關閉，程式結束")
