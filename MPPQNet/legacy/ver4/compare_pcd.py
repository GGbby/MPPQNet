import argparse
import open3d as o3d

def visualize_point_clouds(orig_path, recon_path):
    """
    在同一 3D 空間中，用不同顏色顯示原始點雲與重建點雲。
    """
    # 讀取點雲
    orig = o3d.io.read_point_cloud(orig_path)
    recon = o3d.io.read_point_cloud(recon_path)

    # 指定顏色：原始點雲紅色，重建點雲綠色
    orig.paint_uniform_color([1.0, 0.0, 0.0])  # 紅
    recon.paint_uniform_color([0.0, 1.0, 0.0])  # 綠

    # 建立可視化視窗並繪製
    o3d.visualization.draw_geometries(
        [orig, recon],
        window_name="原始 vs 重建 點雲比較",
        width=800,
        height=600
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可視化原始與重建點雲")
    parser.add_argument("--orig",   required=True, help="原始點雲 .ply 路徑")
    parser.add_argument("--recon",  required=True, help="重建點雲 .ply 路徑")
    args = parser.parse_args()

    visualize_point_clouds(args.orig, args.recon)

