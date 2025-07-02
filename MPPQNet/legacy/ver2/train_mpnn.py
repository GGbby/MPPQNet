# train_mpnn.py
import glob
import numpy as np
import torch
import open3d as o3d
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from networks import PNetwork, FNetwork

def load_all_histograms(ply_folder, num_bins):
    """
    讀取 ply_folder 底下所有 .ply 檔
    1. 用 Open3D 讀 PLY
    2. 透過 build_feature_matrix 產生 F (M×3)
    3. compute_histogram_per_column 得到多個 histogram
    return: Hs (list of numpy arrays)
    """
    Hs = []
    for fn in glob.glob(f"{ply_folder}/*.ply"):
        pcd = o3d.io.read_point_cloud(fn)
        F   = build_feature_matrix(np.asarray(pcd.points))       # Eq.(5)
        hists = compute_histogram_per_column(F, num_bins)        # Method 2.1
        Hs.extend(hists)
    return Hs

def train(Hs, K, epochs=500, lr=1e-3, save_path="mppn_weights.pth"):
    """
    對所有 histogram Hs 進行 P/F network 訓練
    input:
      Hs: list of numpy arrays shape (B,)
      K: 類別數
    return:
      存檔 mppn_weights.pth（包含 pnet & fnet 的 state_dict）
    """
    B = Hs[0].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pnet = PNetwork(input_dim=B, output_dim=K).to(device)
    fnet = FNetwork(input_dim=B, output_dim=K).to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for H in Hs:
            Ht = torch.from_numpy(H).float().to(device)
            # forward
            logits = pnet(Ht)
            p      = torch.softmax(logits, dim=0)
            f_hat  = fnet(Ht)
            # 計算 moments
            levels  = torch.arange(B, dtype=torch.float32, device=device)
            moments = torch.stack([torch.sum((levels**k)*Ht) for k in range(2*K)])
            m_hat   = torch.stack([torch.sum(p * (f_hat**k)) for k in range(2*K)])
            loss    = torch.sum((moments - m_hat)**2)

            # backward
            pnet.optimizer.zero_grad()
            fnet.optimizer.zero_grad()
            loss.backward()
            pnet.optimizer.step()
            fnet.optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, avg loss={total_loss/len(Hs):.6f}")

    # 存權重
    torch.save({'pnet': pnet.state_dict(), 'fnet': fnet.state_dict()}, save_path)
    print("Models saved to", save_path)

def main():
    ply_folder = "path/to/your/ply_folder"
    num_bins   = 64
    K          = 6
    epochs     = 500
    lr         = 1e-3

    print("Loading histograms from PLYs...")
    Hs = load_all_histograms(ply_folder, num_bins)
    print("Total histograms:", len(Hs))

    print("Start training PNetwork & FNetwork...")
    train(Hs, K, epochs, lr, save_path="mppn_weights.pth")

if __name__ == "__main__":
    main()
