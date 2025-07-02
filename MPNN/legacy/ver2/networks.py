# networks.py

import torch
import torch.nn as nn

class PNetwork(nn.Module):
    """
    P-network for moment matching.
    Input: histogram H_tensor (torch.Tensor, shape (B,))
    Output: raw logits for K classes (torch.Tensor, shape (K,))
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=1e-3):
        super(PNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, H_tensor):
        # 直接把 tensor 餵進 model，就不用再從 numpy 轉
        # H_tensor: torch.Tensor shape (B,)
        logits = self.model(H_tensor)
        return logits       # 回傳 tensor

class FNetwork(nn.Module):
    """
    F-network for feature estimation.
    Input: histogram H_tensor (torch.Tensor, shape (B,))
    Output: estimated feature values f_hat (torch.Tensor, shape (K,))
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=1e-3):
        super(FNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, H_tensor):
        # H_tensor: torch.Tensor shape (B,)
        f_hat = self.model(H_tensor)
        return f_hat  # 回傳 tensor
