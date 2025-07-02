# compare_pipelines.py
import numpy as np
import torch
import open3d as o3d
from feature_extraction       import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn                 import run_mppn
from dictionary_and_gaussian  import build_dictionary, compute_gaussian_params
from sklearn.decomposition    import DictionaryLearning

def compute_kmeans_reconstruction(all_feats, centers, labels):
    """
    KMeans 版本的重建：對每個 f_vec_i，用對應 cluster center 重建
    input:
      all_feats: numpy array shape (N_feats, K)
      centers:   numpy array shape (C, K)
      labels:    numpy array shape (N_feats,)
    return:
      all_feats_hat: numpy array shape (N_feats, K)
      mse:           scalar, MSE over all entries
    """
    N_feats, K = all_feats.shape
    all_feats_hat = np.zeros_like(all_feats)
    for i in range(N_feats):
        c = labels[i]
        all_feats_hat[i, :] = centers[c]
    mse = np.mean((all_feats - all_feats_hat)**2)
    return all_feats_hat, mse

def compute_dictlearning_reconstruction(all_feats, codebook_size, dict_alpha, dict_max_iter):
    """
    Eq.(3) 版本 (DictionaryLearning) 的重建：S @ D ≈ all_feats
    input:
      all_feats:    numpy array shape (N_feats, K)
      codebook_size: 整數 C
      dict_alpha:    L1 懲罰權重 μ
      dict_max_iter: 最大迭代次數
    return:
      D:         numpy array shape (C, K)
      S:         numpy array shape (N_feats, C)
      all_feats_hat: numpy array shape (N_feats, K)
      mse:       scalar, MSE over所有 entries
      labels:    numpy array shape (N_feats,), 取每列 S 的 argmax 作 cluster label
    """
    # 1. 初始化 DictionaryLearning（Eq.(3)）
    dict_learner = DictionaryLearning(
        n_components=codebook_size,
        alpha=dict_alpha,
        max_iter=dict_max_iter,
        fit_algorithm='lars',
        transform_algorithm='lasso_lars',
        random_state=0
    )
    # 2. fit 語音 all_feats
    S = dict_learner.fit_transform(all_feats)  # shape (N_feats, C)
    D = dict_learner.components_               # shape (C, K)

    # 3. 重建 all_feats_hat = S @ D
    all_feats_hat = S @ D

    # 4. MSE
    mse = np.mean((all_feats - all_feats_hat)**2)

    # 5. 若要硬編碼 cluster label，可做 argmax(S, axis=1)
    labels = np.argmax(S, axis=1)

    return D, S, all_feats_hat, mse, labels

def compute_compression_ratio(C, K, d):
    """
    根據 Eq.(14) 計算壓縮率 CR = bits(D) / bits(C)
    這裡假設：
      1. 字典 D 以 float32 (32 bits) 存每個 entry
      2. 編碼 C 採「硬編碼索引」：每個向量只存一個 index (log2(C) bits)
    input:
      C: 字典大小 (codebook_size)
      K: f_vec 的維度 (與 Eq.(11) 中 d 相同)
      d: 總共的 f_vec 個數 (N_feats)
    return:
      CR: 壓縮率 (scalar)
    """
    # bits for D = C * K * 32
    bits_D = C * K * 32
    # bits for C_indices = d * log2(C)
    bits_C = d * np.log2(C)
    return bits_D / bits_C

def main():
    # -------------------------------
    # 1. 讀取點雲、建 F、算 histograms、跑 MPPN
    # -------------------------------
    # 讀入 PLY 檔案，假設只有一個 point cloud 檔案
    ply_path = "cloud.ply"  # 請改成你自己的 PLY 路徑
    pcd = o3d.io.read_point_cloud(ply_path)
    pc  = np.asarray(pcd.points)      # M×3

    # 1. Eq.(5)：建 feature matrix F (M×N_dims)
    F = build_feature_matrix(pc)      # M×3 (N_dims=3)

    # 2. Method 2.1：對每一欄 (X,Y,Z) 建 histogram
    num_bins = 64
    hists    = compute_histogram_per_column(F, num_bins)  # list of 3 arrays

    # 3. Method 2.2–2.3：跑 MPPN 得到 p_vec (K,), f_vec (K,), thresholds (K-1)
    K_mppn = 6
    all_f  = []  # 將每個維度的 f_vec 收集起來
    for i, H in enumerate(hists):
        p_vec, f_vec, thresholds = run_mppn(H, K_mppn)
        all_f.append(f_vec)

    # 將所有 f_vec 聚成 all_feats (N_dims × K)
    all_feats = np.vstack(all_f)  # shape (3, K)

    # -------------------------------
    # 2. Pipeline 1: KMeans 字典 + Gaussian 編碼
    # -------------------------------
    # 5. Method 2.4：KMeans 建 dictionary
    codebook_size = 128
    centers, labels_km = build_dictionary(all_feats, codebook_size)  # centers:(128×K), labels: (3,)

    # 6. Eq.(10)：Gaussian 參數估計
    Omegas_km = compute_gaussian_params(all_feats, labels_km, centers)

    # 7. Eq.(11)：encode_points (算 C_codes，但是這裡主要是測 f_vec 重建)
    #    但在實驗中，我們以 dictionary 重建 all_feats(3×K) 做比較
    all_feats_hat_km, mse_km = compute_kmeans_reconstruction(all_feats, centers, labels_km)

    # -------------------------------
    # 3. Pipeline 2: DictionaryLearning (Eq.(3)) + Gaussian 編碼
    # -------------------------------
    dict_alpha    = 1.0   # L1 懲罰權重
    dict_max_iter = 500
    D_dl, S_dl, all_feats_hat_dl, mse_dl, labels_dl = compute_dictlearning_reconstruction(
        all_feats, codebook_size, dict_alpha, dict_max_iter
    )
    # Eq.(10)：Gaussian 參數估計，這裡直接用 labels_dl, D_dl
    Omegas_dl = compute_gaussian_params(all_feats, labels_dl, D_dl)

    # -------------------------------
    # 4. 計算壓縮比 (Eq.(14))：對比兩者的 bits(D) / bits(C)
    #    - bits(D): C * K * 32
    #    - bits(C): d * log2(C)  (d = N_feats = N_dims)
    # -------------------------------
    N_feats = all_feats.shape[0]  # 這裡是 3 (N_dims)
    bitsCR_km = compute_compression_ratio(codebook_size, K_mppn, N_feats)
    bitsCR_dl = compute_compression_ratio(codebook_size, K_mppn, N_feats)

    # -------------------------------
    # 5. 印出比較結果
    # -------------------------------
    print("===== Pipeline Comparison =====")
    print(f"KMeans 版本：")
    print(f"  - all_feats shape        : {all_feats.shape}")
    print(f"  - centers shape          : {centers.shape}")
    print(f"  - KMeans 重建 MSE        : {mse_km:.6f}")
    print(f"  - 壓縮率 CR (bits D / bits C): {bitsCR_km:.2f}")
    print()
    print(f"DictionaryLearning 版本 (Eq.(3))：")
    print(f"  - all_feats shape        : {all_feats.shape}")
    print(f"  - D_dl shape             : {D_dl.shape}")
    print(f"  - DictionaryLearning 重建 MSE: {mse_dl:.6f}")
    print(f"  - 壓縮率 CR (bits D / bits C): {bitsCR_dl:.2f}")
    print("================================")
    print()
    print("注意：重建 MSE 是針對 all_feats (3×K) 上的重建誤差。")

if __name__ == "__main__":
    main()
