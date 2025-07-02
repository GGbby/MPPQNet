# dictionary_and_gaussian.py
import numpy as np
from sklearn.cluster import KMeans

def build_dictionary(f_vecs, num_codewords):
    """
    Method 2.4: 聚所有 {hat f_{i,k}} 為字典 D
    input:
      f_vecs: list of numpy arrays，每個為 shape (K,)
      num_codewords: 字典大小 C
    return:
      centers: numpy array shape (C,K)
      labels: numpy array shape (len(f_vecs),)
    """
    all_feats = np.vstack(f_vecs)  # shape (N_dims, K)
    # KMeans 近似 Eq.(4)：硬分配 VQ
    kmeans = KMeans(n_clusters=num_codewords, random_state=0, n_init=10).fit(all_feats)
    return kmeans.cluster_centers_, kmeans.labels_

def compute_gaussian_params(all_feats, labels, centers):
    """
    Eq.(10): 計算 Gaussian 參數 Omega = { (mu_c, Sigma_c) }
    input:
      all_feats: numpy array shape (N_feats, K)
      labels:    numpy array shape (N_feats,)
      centers:   numpy array shape (C, K)
    return:
      Omegas: list of (mu, cov) tuples，長度 = C
    """
    C, K = centers.shape
    Omegas = []
    for c in range(C):
        mu  = centers[c]
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            cov = np.cov(all_feats[idx].T) + np.eye(K) * 1e-6
        else:
            cov = np.eye(K) * 1e-6
        Omegas.append((mu, cov))
    return Omegas

def encode_points(F, Omegas):
    """
    Eq.(11): 對每個 feature vector f_m 編碼成 codes C
    input:
      F: numpy array shape (M, K)
      Omegas: list of (mu, cov) tuples length = C
    return:
      Cmat: numpy array shape (M, C), Gaussian likelihoods
    """
    M, K = F.shape
    Cnum = len(Omegas)
    Cmat = np.zeros((M, Cnum), dtype=np.float64)
    for c, (mu, cov) in enumerate(Omegas):
        invcov = np.linalg.inv(cov)
        det    = np.linalg.det(cov)
        norm   = 1.0 / np.sqrt((2*np.pi)**K * det)
        diff   = F - mu            # shape (M, K)
        expo   = -0.5 * np.sum(diff @ invcov * diff, axis=1)
        Cmat[:, c] = norm * np.exp(expo)
    return Cmat
