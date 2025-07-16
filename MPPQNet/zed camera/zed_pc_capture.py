import pyzed.sl as sl

def capture_pointcloud(output_path="frame0001.ply"):
    # 1. 初始化參數
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE   # 可視需求改 QUALITY 或 NEURAL
    init.coordinate_units = sl.UNIT.METER

    # 2. 打開相機
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print("❌ 無法開啟 ZED 相機")
        return

    # 3. 抓取一張畫面並讀取點雲
    runtime = sl.RuntimeParameters()
    point_cloud = sl.Mat()
    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # 4. 輸出 PLY 檔
        point_cloud.write(output_path)  # 內建支援 .ply 格式輸出 :contentReference[oaicite:4]{index=4}
        print(f"✅ 已儲存點雲到：{output_path}")

    # 5. 釋放相機
    cam.close()

if __name__ == "__main__":
    capture_pointcloud()
