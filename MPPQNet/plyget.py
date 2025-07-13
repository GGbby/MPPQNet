import pyzed.sl as sl
import numpy as np
import open3d as o3d

def grab_and_save_ply(output_path):
    # 1. 建立 ZED 相機物件
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode       = sl.DEPTH_MODE.ULTRA  # 或你需要的模式
    zed.open(init_params)

    # 2. 拍一張
    mat = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(mat,   sl.MEASURE.XYZRGBA)    # XYZRGBA 會給你 xyzrgb in one
        xyzrgba = mat.get_data()   # shape=(H,W,4) float32 x,y,z,rgbabits
        H, W, _ = xyzrgba.shape

        # 3. 解包成 Open3D point cloud
        pts = xyzrgba[:, :, :3].reshape(-1, 3)
        colors = ((xyzrgba[:, :, 3] & 0xffffff) / (2**24 - 1)).reshape(-1, 1)
        # 如果你能把 0–1 直接抓出來就不用 bit 操作
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(colors, 3, axis=1))

        # 4. 存成 ply
        o3d.io.write_point_cloud(output_path, pcd)

    zed.close()
