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
    GPU 加速版本，包括 thresholds 計算於 GPU 並減少 CPU 轉換。
    如果 pf_weight 不為 None，載入預訓練權重直接 forward；
    否則進行 epochs 次訓練，最後 forward。
    回傳:
      p_vec: numpy array shape (K,), 類別機率向量
      f_vec: numpy array shape (K,), 特徵向量
      thresholds: numpy array shape (K-1,), 相鄰 f_vec 中點
    """
    # 設定裝置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = H.shape[0]
    # 建立網路與優化器，並搬到 device
    pnet = PNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    fnet = FNetwork(input_dim=B, output_dim=K, lr=lr).to(device)
    optimizer_p = pnet.optimizer
    optimizer_f = fnet.optimizer

    # 將 H 轉為 GPU Tensor
    Ht = torch.from_numpy(H).float().to(device)
    # precompute levels 在 GPU 上
    levels = torch.linspace(0, 1, steps=B, device=device)
    # 確定最大矩階
    max_order = max_moment if max_moment is not None else 2 * K
    # loss function
    if use_huber:
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = lambda pred, target: torch.sum((pred - target)**2)

    # 若有預訓練權重：直接 forward
    if pf_weight is not None:
        ckpt = torch.load(pf_weight, map_location=device)
        pnet.load_state_dict(ckpt['pnet'])
        fnet.load_state_dict(ckpt['fnet'])
        pnet.eval(); fnet.eval()
        with torch.no_grad():
            logits = pnet(Ht)
            p_vec_t = torch.softmax(logits, dim=0)
            f_hat_t = fnet(Ht)
            # GPU 上排序並計算 thresholds
            f_sorted, _ = torch.sort(f_hat_t)
            thresh_t = 0.5 * (f_sorted[:-1] + f_sorted[1:])
        return p_vec_t.cpu().numpy(), f_hat_t.cpu().numpy(), thresh_t.cpu().numpy()

    # 無預訓練：訓練迴圈
    pnet.train(); fnet.train()
    for epoch in range(epochs):
        optimizer_p.zero_grad(); optimizer_f.zero_grad()
        logits = pnet(Ht)
        p_vec_t = torch.softmax(logits, dim=0)
        f_hat_t = fnet(Ht)
        # 計算 moment 與 m_hat
        moments = torch.stack([(levels**k * Ht).sum() for k in range(max_order)])
        m_hat = torch.stack([(p_vec_t * (f_hat_t**k)).sum() for k in range(max_order)])
        loss = loss_fn(m_hat, moments)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pnet.parameters(), clip_norm)
        torch.nn.utils.clip_grad_norm_(fnet.parameters(), clip_norm)
        optimizer_p.step(); optimizer_f.step()

    # 訓練後 final forward
    pnet.eval(); fnet.eval()
    with torch.no_grad():
        logits = pnet(Ht)
        p_vec_t = torch.softmax(logits, dim=0)
        f_hat_t = fnet(Ht)
        f_sorted, _ = torch.sort(f_hat_t)
        thresh_t = 0.5 * (f_sorted[:-1] + f_sorted[1:])
    return p_vec_t.cpu().numpy(), f_hat_t.cpu().numpy(), thresh_t.cpu().numpy()
