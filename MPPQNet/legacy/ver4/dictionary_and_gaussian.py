# dictionary_and_gaussian.py

import numpy as np
import torch


def build_dictionary_gpu(f_vecs, num_codewords, iters=50, device=None):
    """
    GPU 加速的 KMeans（取代 sklearn.cluster.KMeans）
    input:
      f_vecs: list of numpy arrays, 每個 shape=(K,)
      num_codewords: int, 字典大小 C
      iters: int, 迭代次數
      device: torch.device 或 None
    return:
      centers: numpy array shape (C, K)
      labels:  numpy array shape (len(f_vecs),)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.vstack(f_vecs).astype(np.float32)         # (N_feats, K)
    X_t = torch.from_numpy(X).to(device)             # (N, K)
    N, K = X_t.shape

    # 隨機初始化 centers
    idx = torch.randperm(N, device=device)[:num_codewords]
    centers = X_t[idx].clone()                       # (C, K)

    for _ in range(iters):
        # pairwise L2 距離
        dists = torch.cdist(X_t, centers)            # (N, C)
        labels = torch.argmin(dists, dim=1)          # (N,)
        # 更新 centers
        new_centers = []
        for c in range(num_codewords):
            mask = (labels == c)
            if mask.sum() == 0:
                new_centers.append(centers[c])
            else:
                new_centers.append(X_t[mask].mean(dim=0))
        centers = torch.stack(new_centers, dim=0)    # (C, K)

    return centers.cpu().numpy(), labels.cpu().numpy()


def compute_gaussian_params_gpu(all_feats, labels, centers, device=None):
    """
    GPU 加速計算每個群的 Gaussian 參數 (mu, Sigma)
    input:
      all_feats: numpy array shape (N_feats, K)
      labels:    numpy array shape (N_feats,)
      centers:   numpy array shape (C, K)
    return:
      Omegas: list of (mu, cov) tuples，長度=C
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(all_feats.astype(np.float32)).to(device)  # (N, K)
    lbl = torch.from_numpy(labels).to(device)                      # (N,)
    C, K = centers.shape

    Omegas = []
    for c in range(C):
        mask = (lbl == c)
        Xc = X[mask]                                             # (Nc, K)
        mu = centers[c]
        if Xc.size(0) > 1:
            # 样本協方差
            Xm = Xc - Xc.mean(dim=0, keepdim=True)               # (Nc, K)
            cov = (Xm.t() @ Xm) / (Xc.size(0) - 1)               # (K, K)
            cov += torch.eye(K, device=device) * 1e-6
        else:
            cov = torch.eye(K, device=device) * 1e-6

        Omegas.append((mu, cov.cpu().numpy()))

    return Omegas


def encode_points(F, Omegas, device=None):
    """
    Eq.(11): 計算每個 feature vector f_m 到所有高斯模型的似然
    input:
      F: numpy array shape (M, K)
      Omegas: list of (mu, cov) tuples length = C
    return:
      Cmat: numpy array shape (M, C)
    """
    M, K = F.shape
    Cnum = len(Omegas)
    F_t = torch.from_numpy(F.astype(np.float32))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_t = F_t.to(device)                                        # (M, K)

    Cmat = torch.zeros(M, Cnum, device=device, dtype=torch.float32)
    const = torch.tensor((2 * np.pi) ** (-0.5 * K), device=device)

    for c, (mu, cov_np) in enumerate(Omegas):
        mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)          # (K,)
        cov_t = torch.from_numpy(cov_np.astype(np.float32)).to(device)    # (K,K)
        invcov = torch.inverse(cov_t)                                     # (K,K)
        det = torch.det(cov_t)
        norm = const * det.pow(-0.5)

        diff = F_t - mu_t                                                  # (M,K)
        expo = -0.5 * torch.sum(diff @ invcov * diff, dim=1)               # (M,)
        Cmat[:, c] = norm * torch.exp(expo)

    return Cmat.cpu().numpy()
