import os
import glob
import numpy as np
import torch
import open3d as o3d
from tqdm import trange, tqdm
from feature_extraction import build_feature_matrix, compute_histogram_per_column
from networks import PNetwork, FNetwork
from torch import nn

def collect_histograms_from_folder_ply(folder, num_bins):
    """
    從資料夾讀取所有 .ply 檔案，
    將點雲轉成 x,y,z,r,g,b 六維特徵後計算直方圖，
    並回傳 histogram 列表。
    """
    Hs = []
    ply_list = sorted(glob.glob(os.path.join(folder, "*.ply")))
    if not ply_list:
        raise RuntimeError(f"在 {folder} 未搜尋到任何 .ply 檔")
    # 總進度條
    pbar = trange(len(ply_list), desc="Overall Progress", position=0)
    # 資訊條（顯示目前檔案）
    info_bar = tqdm(total=0, position=1, bar_format="{desc}")
    # 處理每一個 .ply 檔
    for ply_path in ply_list:
        fname = os.path.basename(ply_path)
        info_bar.set_description(f"處理 {fname}")
        info_bar.refresh()
        # 讀取點雲
        pcd = o3d.io.read_point_cloud(ply_path)
        coords = np.asarray(pcd.points)
        # 座標 min-max 歸一化到 [0,1]
        min_c, max_c = coords.min(axis=0), coords.max(axis=0)
        coords_norm = (coords - min_c) / (max_c - min_c)
        # 顏色轉換到 [0,255]
        colors = np.asarray(pcd.colors) * 255.0
        # 建立六維特徵矩陣 (N,6)
        F = build_feature_matrix(coords_norm, colors)
        # 計算 histogram，回傳 shape=(6,num_bins)
        hists = compute_histogram_per_column(F, num_bins)
        # 展開每維 histogram
        for h in hists:
            Hs.append(h)
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
    save_path
):
    """
    直接從 ply 點雲讀取並計算直方圖，
    使用 PNetwork + FNetwork 進行訓練，
    並依論文方式計算矩及損失，支援梯度裁剪與 Huber Loss。
    """
    # 載入訓練/驗證集直方圖
    print("── 載入訓練集直方圖 …")
    Hs_train = collect_histograms_from_folder_ply(train_ply_folder, num_bins)
    print(f"   共載入 {len(Hs_train)} 條直方圖")
    print("── 載入驗證集直方圖 …")
    Hs_val = collect_histograms_from_folder_ply(val_ply_folder, num_bins)
    print(f"   共載入 {len(Hs_val)} 條直方圖")
    if not Hs_train or not Hs_val:
        raise RuntimeError("訓練/驗證直方圖不足，請確認資料夾內有 ply 檔。")

    # 初始化網路與優化器
    B      = num_bins
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    pnet  = PNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    fnet  = FNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    opt_p, opt_f = pnet.optimizer, fnet.optimizer

    # 損失函數選擇
    loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()

    best_val_loss, best_epoch = float('inf'), -1

    # Epoch 迴圈
    for epoch in trange(epochs, desc="Epochs", unit="epoch"):
        # 訓練模式
        pnet.train(); fnet.train()
        total_train = 0.0
        # levels min-max 正規化到 [0,1]
        levels = torch.arange(B, dtype=torch.float32, device=device)
        levels = (levels - levels.min()) / (levels.max() - levels.min())
        # 處理每一條訓練直方圖
        for H in tqdm(Hs_train, desc="  訓練直方圖", leave=False, unit="hist"):
            Ht = torch.from_numpy(H).float().to(device)
            logits = pnet(Ht)
            p_vec  = torch.softmax(logits, dim=0)
            f_hat  = fnet(Ht)
            MOM    = max_moment if max_moment is not None else 2 * K
            # 計算矩與預測矩
            moments = torch.stack([torch.sum((levels**k) * Ht) for k in range(MOM)])
            m_hat   = torch.stack([torch.sum(p_vec * (f_hat**k)) for k in range(MOM)])
            # 計算損失
            loss = loss_fn(m_hat, moments) if use_huber else torch.sum((moments - m_hat)**2)
            # 反向傳播與梯度裁剪
            opt_p.zero_grad(); opt_f.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pnet.parameters(), clip_norm)
            torch.nn.utils.clip_grad_norm_(fnet.parameters(), clip_norm)
            opt_p.step(); opt_f.step()
            total_train += loss.item()
        avg_train = total_train / len(Hs_train)

        # 驗證模式
        pnet.eval(); fnet.eval()
        total_val = 0.0
        with torch.no_grad():
            for H in tqdm(Hs_val, desc="  驗證直方圖", leave=False, unit="hist"):
                Ht = torch.from_numpy(H).float().to(device)
                logits = pnet(Ht)
                p_vec  = torch.softmax(logits, dim=0)
                f_hat  = fnet(Ht)
                MOM    = max_moment if max_moment is not None else 2 * K
                moments = torch.stack([torch.sum((levels**k) * Ht) for k in range(MOM)])
                m_hat   = torch.stack([torch.sum(p_vec * (f_hat**k)) for k in range(MOM)])
                loss    = loss_fn(m_hat, moments) if use_huber else torch.sum((moments - m_hat)**2)
                total_val += loss.item()
        avg_val = total_val / len(Hs_val)

        # 儲存最佳模型
        if avg_val < best_val_loss:
            best_val_loss, best_epoch = avg_val, epoch
            torch.save({'pnet': pnet.state_dict(), 'fnet': fnet.state_dict()}, save_path)
        tqdm.write(f"Epoch {epoch+1}/{epochs} | 訓練 Loss: {avg_train:.6f} | 驗證 Loss: {avg_val:.6f}")

    print(f"✅ 訓練完成，最佳 驗證 Loss={best_val_loss:.6f} (Epoch {best_epoch+1})")
    print(f"權重已儲存至：{save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="訓練 PNetwork 與 FNetwork 並儲存最佳權重")
    parser.add_argument("--train_ply_folder", type=str, required=True,
                        help="訓練集 ply 檔案資料夾")
    parser.add_argument("--val_ply_folder",   type=str, required=True,
                        help="驗證集 ply 檔案資料夾")
    parser.add_argument("--K", type=int, default=6,
                        help="MPPN 類別數 K")
    parser.add_argument("--num_bins", type=int, default=64,
                        help="直方圖 bin 數")
    parser.add_argument("--epochs", type=int, default=500,
                        help="訓練總 epoch 數")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="學習率")
    parser.add_argument("--clip_norm", type=float, default=1.0,
                        help="梯度裁剪最大范數 (clip_norm)")
    parser.add_argument("--use_huber", action="store_true",
                        help="若啟用，則使用 SmoothL1Loss (Huber Loss)")
    parser.add_argument("--max_moment", type=int, default=None,
                        help="最大矩階數，上限 None 表示 2*K")
    parser.add_argument("--save_path", type=str, default="mppn_pf.pth",
                        help="最佳權重檔案儲存路徑")
    args = parser.parse_args()

    train_pf(
        train_ply_folder=args.train_ply_folder,
        val_ply_folder=args.val_ply_folder,
        K=args.K,
        num_bins=args.num_bins,
        epochs=args.epochs,
        lr=args.lr,
        clip_norm=args.clip_norm,
        use_huber=args.use_huber,
        max_moment=args.max_moment,
        save_path=args.save_path
    )
