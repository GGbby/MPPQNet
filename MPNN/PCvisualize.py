# compare_pcd.py

import argparse
import open3d as o3d

def main():
    parser = argparse.ArgumentParser(
        description="同一視窗對比原始與重建點雲 (紅=原始, 綠=重建)"
    )
    parser.add_argument("--orig",  required=True, help="原始 .ply 檔路徑")
    parser.add_argument("--recon", required=True, help="重建 .ply 檔路徑")
    parser.add_argument("--point_size", type=float, default=2.0, help="點大小")
    args = parser.parse_args()

    # 讀取點雲
    orig_pcd  = o3d.io.read_point_cloud(args.orig)
    recon_pcd = o3d.io.read_point_cloud(args.recon)

    # 上色
    orig_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 紅
    recon_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # 綠

    # 開啟視窗並加入兩個點雲
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Original (Red) vs Reconstructed (Green)",
        width=800, height=600
    )
    vis.add_geometry(orig_pcd)
    vis.add_geometry(recon_pcd)

    # 調整渲染設定
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = [0.1, 0.1, 0.1]

    # 開始互動（旋轉、縮放、平移都很流暢）
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
