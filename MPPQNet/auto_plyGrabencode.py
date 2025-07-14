#!/usr/bin/env python3
# auto_plyGrabencode.py
# 每秒從 ZED 攝影機擷取一張點雲 (.ply)，
# 再呼叫 encode_pipeline.py 壓縮成 .bin，
# 並刪除暫存 .ply

import os
import sys
import time
import subprocess
import numpy as np
import pyzed.sl as sl

# 參數設定
ZED_PLY_DIR   = 'zed_plys'                 # 暫存 .ply 資料夾
BITSTREAM_DIR = 'bitstreams_bin'           # 輸出 .bin 壓縮檔資料夾
PF_WEIGHT     = './pth/b128/mppn_pf_k128b128.pth'  # PFNetwork 權重檔
K             = 128                        # PFNetwork 類別數
NUM_BINS      = 128                        # 直方圖 bins 數
FPS           = 1.0                        # 擷取與壓縮頻率 (Hz)

# 建立資料夾
os.makedirs(ZED_PLY_DIR, exist_ok=True)
os.makedirs(BITSTREAM_DIR, exist_ok=True)

# 將頂點與顏色寫入 ASCII PLY
def write_ply(filename, verts, colors):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(verts, colors):
            x, y, z = v
            r, g, b = (c * 255).astype(np.uint8)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

# 主程序
def main():
    # 1. 初始化 ZED 攝影機
    cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode       = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("[錯誤] 無法開啟 ZED 攝影機")
        sys.exit(1)
    runtime = sl.RuntimeParameters()
    img_mat = sl.Mat()
    depth_mat = sl.Mat()

    last_ts = None

    try:
        while True:
            start = time.time()
            # 2. 擷取
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(1.0 / FPS)
                continue

            # 3. 讀取影像 (LEFT)
            cam.retrieve_image(img_mat, sl.VIEW.LEFT)
            img = img_mat.get_data()  # shape=(H,W,4) 或 (H,W,3)
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]

            # 4. 讀取深度 (XYZ)
            cam.retrieve_measure(depth_mat, sl.MEASURE.XYZ)
            depth = depth_mat.get_data()  # shape=(H,W,3)

            # 5. 過濾並展平
            valid = np.isfinite(depth[:, :, 2]) & (depth[:, :, 2] > 0)
            pts = depth[valid]
            cols = img[valid] / 255.0  # Normalize to [0,1]

            if len(pts) == 0:
                print("[警告] 無有效點雲，跳過本幀")
                time.sleep(1.0 / FPS)
                continue

            # 6. 存成 PLY
            ts = int(time.time())
            ply_path = os.path.join(ZED_PLY_DIR, f"{ts}.ply")
            write_ply(ply_path, pts, cols)

            # 7. 只壓縮新的 PLY
            if ts != last_ts:
                last_ts = ts
                cmd = [
                    sys.executable, 'encode_pipeline.py',
                    '--input_dir', ZED_PLY_DIR,
                    '--pf', PF_WEIGHT,
                    '--latest',
                    '--K', str(K),
                    '--num_bins', str(NUM_BINS),
                    '--out_dir', BITSTREAM_DIR
                ]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"[訊息] 已壓縮 {ts}.ply -> {BITSTREAM_DIR}/{ts}_stream.bin")
                    os.remove(ply_path)
                    print(f"[訊息] 刪除暫存 .ply: {ts}.ply")
                except subprocess.CalledProcessError as e:
                    print(f"[錯誤] 壓縮失敗: {e}")

            # 8. 控制頻率
            elapsed = time.time() - start
            time.sleep(max(0, 1.0 / FPS - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        print("ZED 攝影機已關閉，程式結束")

if __name__ == '__main__':
    main()
