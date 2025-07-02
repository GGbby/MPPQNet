# mpnn_qnn.py
import numpy as np
import torch
from networks import PNetwork, FNetwork

# 請確保在執行前呼叫 torch.manual_seed() 以便 reproducibility

def run_mppn(H, K, epochs=1000, lr=1e-3, device=None):
    """
    Method 2.2 & 2.3: 對 histogram H 執行 MPPN
    input:
      H: numpy array shape (B,), 直方圖機率分佈
      K: 類別數
      epochs: 訓練迭代次數
      lr: 學習率
      device: 'cpu' or 'cuda'
    return:
      p_vec: numpy array shape (K,), 機率向量 {p_{i,k}}
      f_vec: numpy array shape (K,), 特徵向量 {hat f_{i,k}}
      thresholds: numpy array shape (K-1,), 閾值 t_{i,k}
    """
    B = H.shape[0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 networks
    pnet = PNetwork(input_dim=B, output_dim=K).to(device)
    fnet = FNetwork(input_dim=B, output_dim=K).to(device)

    # 將 histogram 轉為 tensor，並移到 device
    Ht = torch.from_numpy(H).float().to(device)
    levels = torch.arange(B, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        # zero_grad
        pnet.optimizer.zero_grad()
        fnet.optimizer.zero_grad()

        # forward
        logits = pnet(Ht)          # shape (K,)
        p      = torch.softmax(logits, dim=0)  # 機率向量
        f_hat  = fnet(Ht)         # shape (K,)

        # Eq.(7): 計算原始矩 m[k]
        moments = torch.stack([torch.sum((levels**k) * Ht) for k in range(2*K)])
        # Eq.(7): 計算估計矩 m_hat[k]
        m_hat   = torch.stack([torch.sum(p * (f_hat**k)) for k in range(2*K)])
        # Eq.(6): loss
        loss    = torch.sum((moments - m_hat)**2)

        # backward & step (一次 backward 讓兩個 network 同時更新)
        loss.backward()
        pnet.optimizer.step()
        fnet.optimizer.step()

    # 訓練完後再做一次 forward 取得最終輸出
    with torch.no_grad():
        logits = pnet(Ht)
        p      = torch.softmax(logits, dim=0)
        f_hat  = fnet(Ht)

    p_vec     = p.cpu().numpy()
    f_vec     = f_hat.cpu().numpy()
    thresholds = np.sort(f_vec)  # Eq.(8): 直接排序得到 (K,) 遞增，再中點取舍

    return p_vec, f_vec, thresholds
