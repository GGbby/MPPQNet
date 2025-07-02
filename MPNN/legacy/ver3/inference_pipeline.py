# inference_pipeline.py 使用CPU

import os
import glob
import argparse
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

from feature_extraction import build_feature_matrix, compute_histogram_per_column
from mpnn_qnn import run_mppn
from dictionary_and_gaussian import compute_gaussian_params

# --------------------
# Torch-based GPU KMeans
# --------------------
class TorchKMeans:
    def __init__(self, n_clusters, n_iter=20, device=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X):
        X = X.to(self.device)
        # 初始化 centroids
        idx = torch.randperm(X.size(0), device=self.device)[:self.n_clusters]
        centroids = X[idx].clone()
        for _ in range(self.n_iter):
            # 計算距離並分配 labels
            dist = torch.cdist(X, centroids)
            labels = dist.argmin(dim=1)
            new_centroids = []
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    new_centroids.append(X[mask].mean(dim=0))
                else:
                    # 空 cluster 隨機重選
                    new_centroids.append(X[torch.randint(0, X.size(0), (1,), device=self.device)].squeeze(0))
            centroids = torch.stack(new_centroids)
        self.centroids = centroids
        return labels.cpu(), centroids.cpu()

# ----------------------------------------
# Torch-based Dictionary Learning
# ----------------------------------------
def torch_dictionary_learning(F_tensor, codebook_size, alpha=1.0, num_iter=200, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_tensor = F_tensor.to(device)
    N, D = F_tensor.shape
    # 初始化 D and S
    D = torch.randn(codebook_size, D, device=device, requires_grad=True)
    S = torch.randn(N, codebook_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([D, S], lr=lr)
    for _ in range(num_iter):
        recon = S @ D
        loss_recon = (F_tensor - recon).pow(2).mean()
        loss_sparse = alpha * S.abs().mean()
        (loss_recon + loss_sparse).backward()
        optimizer.step()
        optimizer.zero_grad()
    D_cpu = D.detach().cpu().numpy()
    S_cpu = S.detach().cpu().numpy()
    labels = S_cpu.argmax(axis=1)
    return D_cpu, S_cpu, labels

# ----------------------------------------
# Main processing function
# ----------------------------------------
def reconstruct_F_from_fvecs_and_thresholds(F, f_vecs, thresholds_list):
    M, N_dims = F.shape
    F_hat = np.zeros_like(F)
    for i in range(N_dims):
        bins = np.digitize(F[:, i], thresholds_list[i])
        F_hat[:, i] = f_vecs[i][bins]
    return F_hat


def process_one_ply(ply_path, pf_weight, method, codebook_size, K_mppn, num_bins, npz_dir, ply_dir):
    # 1. 讀取點雲 & 建 features & histograms
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    F = build_feature_matrix(pts)
    hists = compute_histogram_per_column(F, num_bins)
    # 2. MPPN forward
    all_f, thresholds_list = [], []
    for H in hists:
        _, f_vec, thresholds = run_mppn(H, K_mppn, epochs=1, lr=1e-3, device=None, pf_weight=pf_weight)
        all_f.append(f_vec)
        thresholds_list.append(thresholds)
    all_feats = np.vstack(all_f)

    # 3. 選擇 KMeans or DictLearning
    if method == 'kmeans':
        F_tensor = torch.from_numpy(all_feats).float()
        kmeans = TorchKMeans(n_clusters=codebook_size, n_iter=20)
        labels, centers = kmeans.fit(F_tensor)
        centers_np = centers.numpy()
        Omegas = compute_gaussian_params(all_feats, labels, centers_np)
        all_feats_hat = centers_np[labels]
    else:
        F_tensor = torch.from_numpy(all_feats).float()
        D, S, labels = torch_dictionary_learning(F_tensor, codebook_size, alpha=1.0, num_iter=200, lr=1e-3)
        Omegas = compute_gaussian_params(all_feats, labels, D)
        all_feats_hat = S @ D

    # 4. 重建 F_hat
    F_hat = reconstruct_F_from_fvecs_and_thresholds(F, all_f, thresholds_list)
    base = os.path.splitext(os.path.basename(ply_path))[0]
    # 5. 儲存 npz
    npz_path = os.path.join(npz_dir, f"{base}_{method}_results.npz")
    np.savez(npz_path, all_feats=all_feats, all_feats_hat=all_feats_hat, F_original=F, F_hat=F_hat, codebook_size=codebook_size, K=K_mppn)
    # 6. 輸出 ply
    recon_pcd = o3d.geometry.PointCloud()
    recon_pcd.points = o3d.utility.Vector3dVector(F_hat)
    ply_path_out = os.path.join(ply_dir, f"{base}_{method}_recon.ply")
    o3d.io.write_point_cloud(ply_path_out, recon_pcd)
    return ply_path_out, npz_path


def main(test_dir, pf_weight, method, codebook_size, K_mppn, num_bins, out_dir):
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"{test_dir} 不是有效資料夾")
    npz_dir = os.path.join(out_dir, 'npz')
    ply_dir = os.path.join(out_dir, 'ply')
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)
    ply_files = glob.glob(os.path.join(test_dir, '*.ply'))
    for ply in tqdm(ply_files, desc='Processing', unit='ply'):
        process_one_ply(ply, pf_weight, method, codebook_size, K_mppn, num_bins, npz_dir, ply_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline with Torch GPU')
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--pf', required=True)
    parser.add_argument('--method', choices=['kmeans','dict'], default='kmeans')
    parser.add_argument('--codebook_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--num_bins', type=int, default=64)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    main(args.test_dir, args.pf, args.method, args.codebook_size, args.K, args.num_bins, args.out_dir)
