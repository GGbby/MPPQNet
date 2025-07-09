import os
import glob
import numpy as np
import torch
import open3d as o3d
from tqdm import trange, tqdm
from feature_extraction import build_feature_matrix
from networks import PNetwork, FNetwork
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def collect_histograms_from_folder_ply(folder, num_bins):
    """
    從資料夾讀取所有 .ply 檔案，
    將點雲轉成 x,y,z,r,g,b 六維特徵後在 GPU 上計算直方圖，
    並回傳 histogram 列表。
    """
    Hs = []
    ply_list = sorted(glob.glob(os.path.join(folder, "*.ply")))
    if not ply_list:
        raise RuntimeError(f"在 {folder} 未搜尋到任何 .ply 檔")
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 總進度條
    pbar = trange(len(ply_list), desc="Overall Progress", position=0)
    info_bar = tqdm(total=0, position=1, bar_format="{desc}")
    for ply_path in ply_list:
        fname = os.path.basename(ply_path)
        info_bar.set_description(f"處理 {fname}")
        info_bar.refresh()
        # 讀取點雲並建 F 矩陣
        pcd = o3d.io.read_point_cloud(ply_path)
        coords = np.asarray(pcd.points).astype(np.float32)
        min_c, max_c = coords.min(axis=0), coords.max(axis=0)
        coords_norm = (coords - min_c) / (max_c - min_c)
        colors = (np.asarray(pcd.colors) * 255.0).astype(np.float32)
        F = build_feature_matrix(coords_norm, colors)  # (N,6)
        # 轉成 Tensor 上 GPU
        F_t = torch.from_numpy(F).to(device)
        # 對每個維度計算 GPU histogram
        D = F_t.shape[1]
        for d in range(D):
            col = F_t[:, d]
            min_v = col.min().item()
            max_v = col.max().item()
            hist = torch.histc(col, bins=num_bins, min=min_v, max=max_v)
            hist = hist / hist.sum()
            Hs.append(hist.cpu().numpy().astype(np.float32))
        pbar.update(1)
    pbar.close()
    info_bar.close()
    return Hs


def train_pf(
    train_ply_folder,
    val_ply_folder,
    K,
    num_bins,
    epochs,
    lr,
    clip_norm,
    use_huber,
    max_moment,
    save_path,
    batch_size
):
    # 載入直方圖
    print("── 載入訓練集直方圖 …")
    Hs_train = collect_histograms_from_folder_ply(train_ply_folder, num_bins)
    print(f"   共載入 {len(Hs_train)} 條直方圖")
    print("── 載入驗證集直方圖 …")
    Hs_val = collect_histograms_from_folder_ply(val_ply_folder, num_bins)
    print(f"   共載入 {len(Hs_val)} 條直方圖")
    assert Hs_train and Hs_val, "訓練/驗證直方圖不足"

    # DataLoader
    train_tensor = torch.from_numpy(np.stack(Hs_train)).float()
    val_tensor   = torch.from_numpy(np.stack(Hs_val)).float()
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(val_tensor), batch_size=batch_size,
                              shuffle=False, pin_memory=True)

    # 網路、優化器、裝置
    B = num_bins
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    pnet = PNetwork(input_dim=B, output_dim=K).to(device)
    fnet = FNetwork(input_dim=B, output_dim=K).to(device)
    opt_p, opt_f = pnet.optimizer, fnet.optimizer

    loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()
    best_val_loss, best_epoch = float('inf'), -1

    # 預計算 levels
    levels = torch.linspace(0, 1, steps=B, device=device)
    MOM = max_moment if max_moment is not None else 2 * K

    for epoch in trange(epochs, desc="Epochs", unit="epoch"):
        pnet.train(); fnet.train()
        total_train = 0.0
        for batch in tqdm(train_loader, desc="  訓練批次", leave=False):
            Ht = batch[0].to(device)
            logits = pnet(Ht)
            p_vec = torch.softmax(logits, dim=1)
            f_hat = fnet(Ht)
            # 原始矩與量化矩
            moments = torch.stack([torch.sum((levels**k).unsqueeze(0) * Ht, dim=1)
                                   for k in range(MOM)], dim=1)
            m_hat = torch.stack([torch.sum(p_vec * (f_hat**k), dim=1)
                                  for k in range(MOM)], dim=1)
            loss = loss_fn(m_hat, moments)
            opt_p.zero_grad(); opt_f.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pnet.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(fnet.parameters(), clip_norm)
            opt_p.step(); opt_f.step()
            total_train += loss.item() * Ht.size(0)
        avg_train = total_train / len(train_loader.dataset)

        pnet.eval(); fnet.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  驗證批次", leave=False):
                Ht = batch[0].to(device)
                logits = pnet(Ht)
                p_vec = torch.softmax(logits, dim=1)
                f_hat = fnet(Ht)
                moments = torch.stack([torch.sum((levels**k).unsqueeze(0) * Ht, dim=1)
                                       for k in range(MOM)], dim=1)
                m_hat = torch.stack([torch.sum(p_vec * (f_hat**k), dim=1)
                                      for k in range(MOM)], dim=1)
                loss = loss_fn(m_hat, moments)
                total_val += loss.item() * Ht.size(0)
        avg_val = total_val / len(val_loader.dataset)

        if avg_val < best_val_loss:
            best_val_loss, best_epoch = avg_val, epoch
            torch.save({'pnet': pnet.state_dict(), 'fnet': fnet.state_dict()}, save_path)
        tqdm.write(f"Epoch {epoch+1}/{epochs} | 訓練 Loss: {avg_train:.6f} | 驗證 Loss: {avg_val:.6f}")

    print(f"✅ 訓練完成，最佳 驗證 Loss={best_val_loss:.6f} (Epoch {best_epoch+1})")
    print(f"權重已儲存至：{save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="訓練 PNetwork 與 FNetwork 並儲存最佳權重，支援 GPU histogram")
    parser.add_argument("--train_ply_folder", type=str, required=True)
    parser.add_argument("--val_ply_folder", type=str, required=True)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--use_huber", action="store_true")
    parser.add_argument("--max_moment", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="mppn_pf.pth")
    args = parser.parse_args()

    train_pf(
        args.train_ply_folder, args.val_ply_folder, args.K,
        args.num_bins, args.epochs, args.lr, args.clip_norm,
        args.use_huber, args.max_moment, args.save_path,
        args.batch_size
    )
