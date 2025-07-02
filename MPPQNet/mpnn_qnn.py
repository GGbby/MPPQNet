# mpnn_qnn.py

import numpy as np
import torch
from torch import nn
from networks import PNetwork, FNetwork


def run_mppn(
    H,
    K,
    epochs=1000,
    lr=1e-3,
    device=None,
    pf_weight=None,
    clip_norm=1.0,
    use_huber=False,
    max_moment=None
):
    """
    Method 2.2 & 2.3: 對 histogram H 執行 MPPN。
    如果 pf_weight 不為 None，載入預訓練權重 (.pth) 直接 forward；
    否則跑 epochs 次訓練，最後再 forward。

    參數:
      H: numpy array shape (B,), 直方圖機率分佈
      K: int, 類別數
      epochs: int, 訓練迭代次數
      lr: float, 學習率
      device: torch.device 或 None
      pf_weight: str or None, 預訓練權重檔路徑
      clip_norm: float, 梯度裁剪的最大范數
      use_huber: bool, 是否使用 SmoothL1Loss (Huber Loss)
      max_moment: int or None, 最大矩階數；None 則為 2*K

    回傳:
      p_vec: numpy array shape (K,), 類別機率向量
      f_vec: numpy array shape (K,), 特徵向量
      thresholds: numpy array shape (K-1,), 相鄰 f_vec 中點
    """
    B = H.shape[0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化網路與優化器
    pnet = PNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    fnet = FNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    optimizer_p = pnet.optimizer
    optimizer_f = fnet.optimizer

    # 將 H 轉為 tensor
    Ht = torch.from_numpy(H).float().to(device)

    # levels 做 min-max 正規化
    levels = torch.arange(B, dtype=torch.float32, device=device)
    levels = (levels - levels.min()) / (levels.max() - levels.min())

    # 最大矩階
    max_order = max_moment if max_moment is not None else 2 * K

    # Huber loss 選項
    if use_huber:
        loss_fn = nn.SmoothL1Loss()

    # 若指定預訓練權重，直接載入並 forward
    if pf_weight is not None:
        ckpt = torch.load(pf_weight, map_location=device)
        pnet.load_state_dict(ckpt['pnet'])
        fnet.load_state_dict(ckpt['fnet'])
        pnet.eval()
        fnet.eval()
        with torch.no_grad():
            logits = pnet(Ht)
            p      = torch.softmax(logits, dim=0)
            f_hat  = fnet(Ht)
        p_vec = p.cpu().numpy()
        f_vec = f_hat.cpu().numpy()
        f_sorted   = np.sort(f_vec)
        thresholds = 0.5 * (f_sorted[:-1] + f_sorted[1:])
        return p_vec, f_vec, thresholds

    # 無預訓練模式：訓練迴圈
    for epoch in range(epochs):
        optimizer_p.zero_grad()
        optimizer_f.zero_grad()

        logits = pnet(Ht)
        p_vec_t = torch.softmax(logits, dim=0)
        f_hat   = fnet(Ht)

        # 計算 moments 與 m_hat
        moments = torch.stack([torch.sum((levels**k) * Ht) for k in range(max_order)])
        m_hat   = torch.stack([torch.sum(p_vec_t * (f_hat**k)) for k in range(max_order)])

        if use_huber:
            loss = loss_fn(m_hat, moments)
        else:
            loss = torch.sum((moments - m_hat)**2)

        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(pnet.parameters(), clip_norm)
        torch.nn.utils.clip_grad_norm_(fnet.parameters(), clip_norm)

        optimizer_p.step()
        optimizer_f.step()

    # 最後再 forward 一次
    with torch.no_grad():
        logits = pnet(Ht)
        p_vec_t = torch.softmax(logits, dim=0)
        f_hat   = fnet(Ht)
    p_vec = p_vec_t.cpu().numpy()
    f_vec = f_hat.cpu().numpy()
    f_sorted   = np.sort(f_vec)
    thresholds = 0.5 * (f_sorted[:-1] + f_sorted[1:])
    return p_vec, f_vec, thresholds
