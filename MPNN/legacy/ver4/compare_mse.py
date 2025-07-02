# compare_mse.py

import numpy as np

def compute_mse(a, b):
    """
    計算 MSE：np.mean((a - b)**2)
    """
    return np.mean((a - b)**2)

def main():
    """
    這支程式假設目前工作目錄底下有：
      - {base}_kmeans_results.npz
      - {base}_dictlearning_results.npz
    其中 NPZ 檔案內含：
      all_feats, all_feats_hat, F_original, F_hat, codebook_size, K
    輸入參數：--base sceneA
    就會讀 sceneA_kmeans_results.npz & sceneA_dictlearning_results.npz，
    計算 all_feats/F 的 MSE，以及理論壓縮率 CR=bits(D)/bits(C)。
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True,
                        help="檔名前綴，例如 --base sceneA，會讀 sceneA_kmeans_results.npz 及 sceneA_dictlearning_results.npz")
    args = parser.parse_args()
    base = args.base

    # ---- 讀 KMeans 結果
    km_npz  = f"{base}_kmeans_results.npz"
    data_km = np.load(km_npz)
    af_km   = data_km["all_feats"]         # shape (3, K)
    afh_km  = data_km["all_feats_hat"]     # shape (3, K)
    F_km    = data_km["F_original"]        # shape (M, 3)
    Fh_km   = data_km["F_hat"]             # shape (M, 3)
    C_km    = data_km["codebook_size"].item()
    K_mppn  = data_km["K"].item()

    # ---- 讀 DictLearning 結果
    dl_npz  = f"{base}_dictlearning_results.npz"
    data_dl = np.load(dl_npz)
    af_dl   = data_dl["all_feats"]
    afh_dl  = data_dl["all_feats_hat"]
    F_dl    = data_dl["F_original"]
    Fh_dl   = data_dl["F_hat"]
    C_dl    = data_dl["codebook_size"].item()
    K_mppn2 = data_dl["K"].item()

    # ---- 計算 MSE
    mse_af_km = compute_mse(af_km, afh_km)
    mse_F_km  = compute_mse(F_km, Fh_km)

    mse_af_dl = compute_mse(af_dl, afh_dl)
    mse_F_dl  = compute_mse(F_dl, Fh_dl)

    # ---- 理論壓縮率 (bits(D)/bits(C))
    # bits(D) = K * log2(C). bits(C) = M * log2(K). CR = (K*log2(C))/(M*log2(K))
    M_points = F_km.shape[0]  # M
    bitsD_km = K_mppn * np.log2(C_km)
    bitsC_km = M_points * np.log2(K_mppn)
    CR_km    = bitsD_km / bitsC_km

    bitsD_dl = K_mppn2 * np.log2(C_dl)
    bitsC_dl = M_points * np.log2(K_mppn2)
    CR_dl    = bitsD_dl / bitsC_dl

    # ---- 印出比較結果
    print("===== Compare all_feats/F MSE & Theoretical CR =====")
    print(f"KMeans:      all_feats MSE = {mse_af_km:.6f},   F MSE = {mse_F_km:.6f},   CR = {CR_km:.4f}")
    print(f"DictLearning: all_feats MSE = {mse_af_dl:.6f},   F MSE = {mse_F_dl:.6f},   CR = {CR_dl:.4f}")

if __name__ == "__main__":
    main()
