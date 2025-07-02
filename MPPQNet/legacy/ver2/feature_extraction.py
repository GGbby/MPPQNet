# feature_extraction.py
import numpy as np

def build_feature_matrix(point_cloud):
    """
    Eq.(5): 從點雲 Γ 建出 feature matrix F (M×N)。
    這裡直接使用 XYZ 座標作為特徵，
    可擴展為 normals、curvature 等。
    input:
      point_cloud: M×3 numpy array
    return:
      F: M×3 numpy array
    """
    return point_cloud.copy()

def compute_histogram_per_column(F, num_bins, value_range=None):
    """
    Method 2.1: 對 feature matrix F 的每個欄位 i 建 histogram H_i
    input:
      F: M×N numpy array
      num_bins: 每 histogram bin 數 b_i
      value_range: (min,max) 直方圖範圍，預設為該欄位 min,max
    return:
      hist_list: list of N numpy arrays, shape (num_bins,)
    """
    M, N = F.shape
    hist_list = []
    for i in range(N):
        col = F[:, i]
        vmin, vmax = (value_range if value_range is not None else (col.min(), col.max()))
        hist, _ = np.histogram(col, bins=num_bins, range=(vmin, vmax), density=False)
        hist = hist.astype(np.float64)
        hist /= hist.sum()
        hist_list.append(hist)
    return hist_list
