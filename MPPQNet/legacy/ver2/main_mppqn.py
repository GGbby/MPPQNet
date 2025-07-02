# main_mppqn.py
import numpy as np
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn import run_mppn
from dictionary_and_gaussian import build_dictionary, compute_gaussian_params, encode_points

def main(point_cloud, num_bins, K, num_codewords):
    """
    pipeline:
      1. Eq.(5): build feature matrix F
      2. Method 2.1: histogram per column
      3. Method 2.2–2.3: run MPPN (P/F network) 得到 p_vec, f_vec, thresholds
      4. Eq.(9): super-point 分群 (np.digitize)
      5. Method 2.4: KMeans 建 dictionary
      6. Eq.(10): Gaussian 參數估計
      7. Eq.(11): encode points
    return:
      C: numpy array shape (M, Cnum)
      centers: numpy array shape (Cnum, K)
      Omegas: list of (mu, cov)
      SP: list of length N_dims，每個元素是長度 K 的 list (super-point 子集合)
    """
    # 1. Eq.(5): build feature matrix
    F = build_feature_matrix(point_cloud)  # M×N_dims

    # 2. Method 2.1: histogram per column
    hists = compute_histogram_per_column(F, num_bins)

    all_f = []
    SP = []  # super-point 分群結果

    # 3. Method 2.2–2.3: 對每個維度執行 MPPN
    for i, H in enumerate(hists):
        p_vec, f_vec, thresholds = run_mppn(H, K)
        all_f.append(f_vec)

        # 4. Eq.(9): super-point 分群
        bins = np.digitize(F[:, i], thresholds)
        SP_i = [F[bins == k] for k in range(K)]
        SP.append(SP_i)

    # 5. Method 2.4: KMeans 建 dictionary D
    all_feats = np.vstack(all_f)  # shape (N_dims, K)
    centers, labels = build_dictionary(all_feats, num_codewords)

    # 6. Eq.(10): Gaussian 參數估計
    Omegas = compute_gaussian_params(all_feats, labels, centers)

    # 7. Eq.(11): encode points
    C = encode_points(F, Omegas)

    return C, centers, Omegas, SP

if __name__ == '__main__':
    # 範例：讀 cloud.xyz（如果你改用 .ply，請改成 open3d 讀取）
    pc = np.loadtxt('cloud.xyz')
    C, D, Omegas, SP = main(pc, num_bins=64, K=6, num_codewords=128)
    print('C shape:', C.shape)
    print('D shape:', D.shape)
    print('Ω count:', len(Omegas))
    print('Super-point groups per dimension:', len(SP))
