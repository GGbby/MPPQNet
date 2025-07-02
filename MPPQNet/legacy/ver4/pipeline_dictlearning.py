# pipeline_dictlearning.py

import argparse
import os
import glob
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from feature_extraction      import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn                import run_mppn
from dictionary_and_gaussian import compute_gaussian_params

def dict_learning_gpu(X, codebook_size, alpha, max_iter, lr=1e-3, device=None):
    """
    PyTorch GPU Dictionary Learning via joint SGD:
      minimize ||X - S D||^2 + alpha * ||S||_1
    input:
      X: numpy array shape (N_feats, K)
    return:
      D_np: numpy array shape (codebook_size, K)
      S_np: numpy array shape (N_feats, codebook_size)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float().to(device)
    N, K = X_t.shape
    D = torch.randn(codebook_size, K,   requires_grad=True, device=device)
    S = torch.randn(N,             codebook_size, requires_grad=True, device=device)
    opt = torch.optim.Adam([D, S], lr=lr)
    for _ in range(max_iter):
        recon = S @ D
        loss  = ((X_t - recon)**2).mean() + alpha * S.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return D.detach().cpu().numpy(), S.detach().cpu().numpy()

def reconstruct_F_from_fvecs_and_labels(F, all_feats_hat_dl):
    """
    Eq.(12) 硬編碼重建 (DictionaryLearning 版)
    input:
      F: numpy array shape (M, N_dims)
      all_feats_hat_dl: numpy array shape (N_dims, K)
    return:
      F_hat: numpy array shape (M, N_dims)
    """
    M, N_dims = F.shape
    F_hat = np.zeros_like(F)
    for i in range(N_dims):
        idx = np.argmin(np.abs(F[:, i][:, None] - all_feats_hat_dl[i][None, :]), axis=1)
        F_hat[:, i] = all_feats_hat_dl[i][idx]
    return F_hat

def main(input_dir, pf_weight=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置：", device)
    ply_list = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    if not ply_list:
        print(f"找不到任何 .ply 檔於 {input_dir}")
        return

    # 建立儲存資料夾
    npz_dir = os.path.join("results", "npz")
    ply_dir = os.path.join("results", "ply")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # 批次處理所有 .ply，單一 tqdm 進度條
    for ply_path in tqdm(ply_list, desc="DictLearning pipeline", unit="file"):
        base = os.path.splitext(os.path.basename(ply_path))[0]

        # 1. 讀取點雲 → build F → histograms
        pcd = o3d.io.read_point_cloud(ply_path)
        pc  = np.asarray(pcd.points)
        F   = build_feature_matrix(pc)
        hists = compute_histogram_per_column(F, 64)

        # 2. MPPN forward
        all_f = []
        for H in hists:
            _, f_vec, _ = run_mppn(H, 6, epochs=1, lr=1e-3, device=device, pf_weight=pf_weight)
            all_f.append(f_vec)
        all_feats = np.vstack(all_f)

        # 3. GPU Dictionary Learning + Gaussian
        D_dl, S_dl = dict_learning_gpu(all_feats, 128, alpha=1.0, max_iter=500, lr=1e-3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels_dl  = np.argmax(S_dl, axis=1)
        _ = compute_gaussian_params(all_feats, labels_dl, D_dl)

        # 4. 重建 F_hat
        all_feats_hat = S_dl @ D_dl
        F_hat_dl      = reconstruct_F_from_fvecs_and_labels(F, all_feats_hat)

        # 5. 儲存 .npz + 重建 .ply
        np.savez(
            os.path.join(npz_dir, f"{base}_dictlearning.npz"),
            all_feats=all_feats,
            all_feats_hat=all_feats_hat,
            F_original=F,
            F_hat=F_hat_dl
        )
        recon = o3d.geometry.PointCloud()
        recon.points = o3d.utility.Vector3dVector(F_hat_dl)
        o3d.io.write_point_cloud(os.path.join(ply_dir, f"{base}_dictlearning.ply"), recon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch GPU DictionaryLearning pipeline")
    parser.add_argument("--input_dir", required=True, help="資料夾，含多個 .ply 檔")
    parser.add_argument("--pf",         default=None,  help="可選：mppn_pf.pth")
    args = parser.parse_args()
    main(args.input_dir, args.pf)
