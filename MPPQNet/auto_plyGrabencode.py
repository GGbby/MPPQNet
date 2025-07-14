import time
import os
import sys
import subprocess

import numpy as np
import open3d as o3d
import pyzed.sl as sl

# —— 參數設定 ——#
ZED_PLY_DIR   = 'zed_plys'                 # 暫存 .ply 檔資料夾
BITSTREAM_DIR = 'bitstreams_bin'           # 輸出 .bin 壓縮檔資料夾
PF_WEIGHT     = './pth/b128/mppn_pf_k128b128.pth'  # PFNetwork 權重檔
K             = 128                        # PFNetwork 類別數
NUM_BINS      = 128                        # 直方圖 bins 數
FPS           = 1.0                        # 擷取與壓縮頻率 (Hz)，1 次／秒
#—————————————#

# 建立資料夾
os.makedirs(ZED_PLY_DIR,  exist_ok=True)
os.makedirs(BITSTREAM_DIR, exist_ok=True)

# 1. 只開啟一次 ZED 攝影機
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode       = sl.DEPTH_MODE.ULTRA
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("[錯誤] ZED 相機開啟失敗")
    sys.exit(1)
runtime = sl.RuntimeParameters()
mat     = sl.Mat()

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
        zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
        xyzrgba = mat.get_data()            # shape=(H,W,4), float32
        H, W, _ = xyzrgba.shape

        # 4. 展平並拆出 三維座標 與 顏色
        pts = xyzrgba[..., :3].reshape(-1, 3)             # (H*W, 3)
        rgba_flat = xyzrgba[..., 3].reshape(-1).astype(np.uint32)  # (H*W,)
        rgb_int   = rgba_flat & 0x00FFFFFF                # 取低 24-bit 作為 RGB
        # normalize 到 [0,1]
        cols = (rgb_int.astype(np.float32) / (2**24 - 1)).reshape(-1,1)
        cols = np.repeat(cols, 3, axis=1)                 # (H*W, 3)

        # 5. 過濾掉 z<=0 或 非有限值
        mask = (pts[:,2] > 0) & np.isfinite(pts).all(axis=1)
        pts, cols = pts[mask], cols[mask]

        # 6. 輸出暫存 .ply
        ts = int(time.time())
        ply_path = os.path.join(ZED_PLY_DIR, f"{ts}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
        o3d.io.write_point_cloud(ply_path, pcd)

        # 7. 只對「新的時間戳」執行壓縮
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
                print(f"[訊息] 已刪除暫存 ply：{ts}.ply")
            except subprocess.CalledProcessError as e:
                print(f"[錯誤] 壓縮失敗: {e}")

        # 8. 控制擷取/壓縮頻率
        elapsed = time.time() - start
        time.sleep(max(0, (1.0 / FPS) - elapsed))

except KeyboardInterrupt:
    # 使用者中斷 (Ctrl+C) 時，正常關閉相機
    pass
finally:
    zed.close()
    print("ZED 已關閉，程式結束")