# auto_plyGrabencode.py (改良版)
# 正常自動化流程不會用到，測試用
import time, os, subprocess, sys
from plyget import grab_and_save_ply

ZED_PLY_DIR   = 'zed_plys'
BITSTREAM_DIR = 'bitstreams_bin'
os.makedirs(ZED_PLY_DIR,   exist_ok=True)
os.makedirs(BITSTREAM_DIR,  exist_ok=True)

last_ts = None
while True:
    ts = int(time.time())
    ply_path = os.path.join(ZED_PLY_DIR, f"{ts}.ply")
    try:
        grab_and_save_ply(ply_path)
    except Exception as e:
        print(f"[Error] grab_and_save_ply(): {e}")
        time.sleep(1.0)
        continue

    # 只在秒數跳變時觸發一次
    if ts != last_ts:
        last_ts = ts
        cmd = [
            sys.executable, 'encode_pipeline.py',
            '--input_dir',  ZED_PLY_DIR,
            '--pf',         './pth/b128/mppn_pf_k128b128.pth',
            '--latest',
            '--K',           '128',
            '--num_bins',    '128',
            '--num_clusters','128',
            '--num_res_clusters','128',
            '--out_dir',     BITSTREAM_DIR,
            '--alpha',       '0.7'
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] encode failed: {e}")

    time.sleep(1.0)
