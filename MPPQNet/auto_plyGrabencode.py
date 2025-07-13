# auto_plyGrabencode.py
# 從 ZED 攝影機擷取最新一幀 .ply，呼叫 encode_pipeline.py 壓縮成 .bin，然後刪除該 .ply

import time
import os
import subprocess
import sys
from plyget import grab_and_save_ply

#—— 參數設定 ——#
ZED_PLY_DIR   = 'zed_plys'       # 存放擷取的 ply 檔案資料夾
BITSTREAM_DIR = 'bitstreams_bin'  # 輸出 .bin 壓縮檔資料夾
PF_WEIGHT     = './pth/b128/mppn_pf_k128b128.pth'  # PFNetwork 權重檔路徑
K             = 128               # PFNetwork 類別數 K
NUM_BINS      = 128               # histogram bins 數量
FPS           = 1.0               # 擷取與壓縮頻率 (Hz)
#—————————————#

os.makedirs(ZED_PLY_DIR,   exist_ok=True)
os.makedirs(BITSTREAM_DIR,  exist_ok=True)

last_ts = None
while True:
    ts = int(time.time())
    ply_path = os.path.join(ZED_PLY_DIR, f"{ts}.ply")

    # 1. 擷取並儲存 ply
    try:
        grab_and_save_ply(ply_path)
    except Exception as e:
        print(f"[錯誤] grab_and_save_ply(): {e}")
        time.sleep(1.0 / FPS)
        continue

    # 2. 只在新的時間戳才進行壓縮
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
            print(f"[訊息] 已壓縮 {ts}.ply -> bitstreams_bin/{ts}_stream.bin")
            # 3. 刪除剛剛壓縮過的 ply
            os.remove(ply_path)
            print(f"[訊息] 已刪除暫存 ply：{ts}.ply")
        except subprocess.CalledProcessError as e:
            print(f"[錯誤] 壓縮失敗: {e}")

    # 4. 等待下個週期
    time.sleep(1.0 / FPS)
