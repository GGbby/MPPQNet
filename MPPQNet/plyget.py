import pyzed.sl as sl
import numpy as np
import open3d as o3d

def grab_and_save_ply(output_path):
    # 1. 建立 ZED 相機物件（只做一次即可，也可移到外層）
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode       = sl.DEPTH_MODE.ULTRA
    zed.open(init_params)

    # 2. 拍一張
    mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # 同時取 XYZ 與 RGBA
        zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
        xyzrgba = mat.get_data()  # float32, shape=(H,W,4)

        # 3. 解包成 Open3D point cloud
        H, W, _ = xyzrgba.shape
        # 取前三維 (XYZ)，如果未來 MEASURE 改版有多維的話也不會多 unpack
        pts = xyzrgba[..., :3].reshape(-1, 3)

        # 先轉整數，再做位元運算
        rgba_float = xyzrgba[:, :, 3]            # float32 代表 RGBA bits
        rgba_int   = rgba_float.astype(np.uint32)
        # 取下 24 位元 (R|G|B)，並 normalize 到 [0,1]
        rgb_int    = rgba_int & 0x00FFFFFF
        colors     = (rgb_int.astype(np.float32) / float(2**24 - 1)).reshape(-1, 1)
        # 重複成 3 channel
        colors_rgb = np.repeat(colors, 3, axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        # 4. 存成 PLY
        o3d.io.write_point_cloud(output_path, pcd)

    zed.close()
