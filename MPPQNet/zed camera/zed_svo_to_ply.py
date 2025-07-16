#!/usr/bin/env python3
import argparse
import pyzed.sl as sl
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="從 ZED SVO 檔擷取點雲並輸出 PLY 檔")
    p.add_argument("svo_path", help="輸入的 .svo 檔路徑")
    p.add_argument(
        "-o", "--output",
        default="output.ply",
        help="輸出 PLY 檔（預設: output.ply）"
    )
    p.add_argument(
        "-s", "--skip",
        type=int, default=60,
        help="跳過前 N 幀熱身深度演算（預設: 60）"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1. 初始化參數
    init_params = sl.InitParameters()
    init_params.svo_input_filename = args.svo_path
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    # 2. 打開 SVO 檔
    cam = sl.Camera()
    err = cam.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"❌ 無法開啟 SVO 檔: {args.svo_path} （錯誤代碼: {err}）")
        sys.exit(1)

    runtime_params = sl.RuntimeParameters()

    # 3. 丟掉前幾幀（讓深
