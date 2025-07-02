import torch
import numpy as np

def kmeans_gpu(X, num_clusters, iters=50, device=None):
    """
    PyTorch GPU KMeans
    input:
      X: numpy array shape (N, D)
      num_clusters: int, 群集數量
      iters: int, 迭代次數
    return:
      labels: numpy array shape (N,)
      centers: numpy array shape (C, D)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)  # (N, D)
    N, D = X_t.shape
    if num_clusters > N:
        num_clusters = N
    # 隨機初始化 centers
    idx = torch.randperm(N, device=device)[:num_clusters]
    centers = X_t[idx].clone()  # (C, D)
    # 迭代更新
    for _ in range(iters):
        dists = torch.cdist(X_t, centers)   # (N, C)
        labels = torch.argmin(dists, dim=1) # (N,)
        new_centers = []
        for k in range(num_clusters):
            pts_k = X_t[labels == k]
            if pts_k.numel() == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(pts_k.mean(dim=0))
        centers = torch.stack(new_centers, dim=0)
    return labels.cpu().numpy(), centers.cpu().numpy()