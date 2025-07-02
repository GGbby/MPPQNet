import numpy as np
import cv2

def compute_histogram_and_moments(image, max_moment):
    """
    計算影像的直方圖與各階矩。
    input:
      image: 2D numpy array, 灰階影像
      max_moment: int, 最大矩階數 (通常為 2N-1)
    output:
      hist: 長度256的一維整數陣列
      moments: 長度 max_moment+1 的浮點陣列，m0, m1, ..., m_max_moment
    """
    # 計算直方圖
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()  # 正規化為機率分佈
    # 計算矩
    levels = np.arange(256)
    moments = np.array([np.sum((levels ** k) * hist) for k in range(max_moment + 1)])
    return hist, moments

def moment_preserving_threshold(image, N, alpha_P=0.1, alpha_Z=0.1, tol=1e-6, max_iter=500):
    """
    以論文方法對影像進行 N 級矩保存閾值分割。
    input:
      image: 灰階影像 2D numpy array
      N: 類別數 (threshold 等於 N-1)
      alpha_P: 更新 P-network 的學習率
      alpha_Z: 更新 Z-network 的學習率
      tol: 收斂容限
      max_iter: 最大迭代次數
    output:
      thresholds: 長度 N-1 的閾值列表
    """
    # 1. 計算直方圖與矩
    hist, moments = compute_histogram_and_moments(image, max_moment=2*N-1)
    
    # 2. 初始化 Z_i 為灰度等距
    Z = np.linspace(0, 255, N, dtype=np.float64)
    # 3. 隨機初始化 P_i (正規化)
    P = np.ones(N, dtype=np.float64) / N
    
    # 迭代求解
    for it in range(max_iter):
        # 計算預估矩 m_k_hat
        m_hat = np.array([np.sum(P * (Z ** k)) for k in range(2*N)])
        
        # 計算誤差與梯度
        error = m_hat - moments[:2*N]
        
        # P-network 更新 dE/dP_i = 2 * sum_k error_k * Z_i^k
        grad_P = 2 * np.array([np.sum(error * (Z[i] ** np.arange(2*N))) for i in range(N)])
        P_new = P - alpha_P * grad_P
        P_new = np.clip(P_new, 1e-8, 1.0)  # 防止越界
        P_new /= P_new.sum()  # 重新正規化
        
        # Z-network 更新 dE/dZ_i = 2 * sum_k error_k * P_i * k * Z_i^(k-1)
        grad_Z = 2 * np.array([np.sum(error[1:] * P[i] * np.arange(1, 2*N) * (Z[i] ** (np.arange(1, 2*N)-1))) for i in range(N)])
        Z_new = Z - alpha_Z * grad_Z
        Z_new = np.clip(Z_new, 0, 255)
        
        # 檢查收斂
        if np.linalg.norm(P_new - P) < tol and np.linalg.norm(Z_new - Z) < tol:
            break
        
        P, Z = P_new, Z_new
    
    # 閾值為相鄰 Z 的中點
    thresholds = [(Z[i] + Z[i+1]) / 2 for i in range(N-1)]
    return thresholds

# 範例使用
if __name__ == "__main__":
    img = cv2.imread("input_image.png", cv2.IMREAD_GRAYSCALE)
    thresholds = moment_preserving_threshold(img, N=3)
    print("計算得到的多級閾值:", thresholds)

