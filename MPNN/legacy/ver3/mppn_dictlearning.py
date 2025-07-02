# mppn_dictlearning.py
import numpy as np
import torch
from sklearn.decomposition import DictionaryLearning
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn import run_mppn
from dictionary_and_gaussian import compute_gaussian_params, encode_points

def mppn_with_dictlearning(point_cloud, num_bins, K, codebook_size, dict_alpha, dict_max_iter=1000):
    """
    用 MPPN 先產生 f_vecs，接著以 Eq.(3) DictionaryLearning 替代 KMeans
    input:
      point_cloud: M×3 numpy array
      num_bins: histogram bin 數
      K:        MPPN 中每維度的量化類別數
      codebook_size: Eq.(3) 中字典的大小 C
      dict_alpha:     Eq.(3) 中 L1 惩罰權重 μ
      dict_max_iter:  DictionaryLearning 最大迭代次數
    return:
      D:         numpy array shape (C, K) 字典中心
      S:         numpy array shape (N_feats, C) 稀疏編碼矩陣
      Omegas:    list of (mu, cov) tuples
      C_codes:   numpy array shape (M, C) Gaussian likelihood（與 encode_points 一致）
      all_f:     list of f_vec (length = N_dims)
    """
    # 1. Eq.(5): 建 feature matrix F
    F = build_feature_matrix(point_cloud)  # shape (M, N_dims)

    # 2. Method 2.1: histogram per column
    hists = compute_histogram_per_column(F, num_bins)

    all_f = []
    SP    = []  # super-point（此版本不著重 super-point，只記錄 f_vec）
    for i, H in enumerate(hists):
        # 3. Method 2.2–2.3: run MPPN 得到 f_vec
        p_vec, f_vec, thresholds = run_mppn(H, K)
        all_f.append(f_vec)

        # 4. Optionally: super-point 分群 (此處可自行加入 np.digitize)
        # bins = np.digitize(F[:, i], thresholds)
        # SP_i = [F[bins == k] for k in range(K)]
        # SP.append(SP_i)

    # 將所有 f_vecs 串成 all_feats (shape = N_dims × K)
    all_feats = np.vstack(all_f)  # each row is one f_vec, 共 N_dims rows

    # 5. Eq.(3): DictionaryLearning 帶 L1 懲罰
    #    minimize ||all_feats - S D||^2 + dict_alpha * ||S||_1
    # sklearn 的 DictionaryLearning 回傳 components_ = D (C×K)
    dict_learner = DictionaryLearning(
        n_components=codebook_size,
        alpha=dict_alpha,
        max_iter=dict_max_iter,
        fit_algorithm='lars',
        transform_algorithm='lasso_lars',
        random_state=0
    )
    S = dict_learner.fit_transform(all_feats)  # shape (N_feats, C)
    D = dict_learner.components_               # shape (C, K)

    # 6. Eq.(10): 用 all_feats, S, D 估計 Gaussian parameters
    #    這裡以 DictionaryLearning 方式做 code，不再對每個 cluster 做力扣
    labels = np.argmax(S, axis=1)  # 把最強 coefficient 的 index 當成 cluster label
    Omegas = compute_gaussian_params(all_feats, labels, D)

    # 7. Eq.(11): encode points
    #    用原始 F 產生 code C_codes (M, C)
    C_codes = encode_points(F, Omegas)

    return D, S, Omegas, C_codes, all_f

if __name__ == '__main__':
    # 範例：讀 cloud.xyz
    pc = np.loadtxt('cloud.xyz')
    num_bins      = 64
    K             = 6
    codebook_size = 128
    dict_alpha    = 1.0    # Eq.(3) 中 L1 懲罰強度，可調整
    dict_max_iter = 500

    D, S, Omegas, C_codes, all_f = mppn_with_dictlearning(
        pc, num_bins, K, codebook_size, dict_alpha, dict_max_iter
    )
    print("DictionaryLearning D shape:", D.shape)
    print("Sparse codes S shape:", S.shape)
    print("Ω length:", len(Omegas))
    print("Encoded C_codes shape:", C_codes.shape)

    # 這裡可額外計算重建誤差或壓縮率，比較與 KMeans 流程的差異
