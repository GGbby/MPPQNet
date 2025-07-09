import numpy as np
import torch

def build_feature_matrix(point_cloud, colors=None):
    """
    Eq.(5): 從點雲 Γ 建出 feature matrix F (M×N)。
    輸出 numpy array，與原版一致。
    """
    pts = point_cloud.astype(np.float32)
    if colors is None:
        return pts.copy()
    cols = (colors * 255.0).astype(np.float32)
    return np.hstack([pts, cols])


def compute_histogram_per_column(F, num_bins, value_range=None, use_gpu=True):
    """
    Method 2.1: 對 feature matrix F 的每個欄位建 histogram H_i，GPU 加速版本。
    input:
      F: M×N numpy array
      num_bins: 每 histogram bin 數 b_i
      value_range: (min,max) 直方圖範圍，預設由資料決定
      use_gpu: 是否使用 GPU 加速 (默認 True)
    return:
      hist_list: list of N numpy arrays, shape (num_bins,)
    """
    # 選擇裝置
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    # 建立 GPU Tensor
    F_t = torch.from_numpy(F.astype(np.float32)).to(device)
    hist_list = []
    dims = F_t.shape[1]
    for i in range(dims):
        col = F_t[:, i]
        # 取最小/最大值
        if value_range is None:
            vmin, vmax = float(col.min()), float(col.max())
        else:
            vmin, vmax = value_range
        # GPU histogram
        hist = torch.histc(col, bins=num_bins, min=vmin, max=vmax)
        hist = hist / hist.sum()
        # 回 CPU 並轉為 numpy
        hist_list.append(hist.cpu().numpy().astype(np.float64))
    return hist_list
