import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

class DDFM(nn.Module):
    def __init__(self, dim=512, tau=0.1):
        super(DDFM, self).__init__()
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, des_tensor):
        N, dim = x.shape
        n, _ = des_tensor.shape

        # 语义相似性计算
        des_tensor = des_tensor.to(dtype=torch.float32)
        S = torch.matmul(x, des_tensor.t()) / self.tau
        A = F.softmax(S, dim=-1)

        # 动态聚类
        des_sim = F.cosine_similarity(des_tensor.unsqueeze(1), des_tensor.unsqueeze(0), dim=-1)
        des_sim_clipped = torch.clamp(des_sim, max=1.0)
        distance_matrix = 1 - des_sim_clipped.cpu().numpy()
        clustering = DBSCAN(eps=0.5, min_samples=1, metric='precomputed').fit(distance_matrix)
        labels = torch.tensor(clustering.labels_, device=x.device)
        K = labels.max() + 1
        groups = [torch.where(labels == k)[0] for k in range(K)]

        # 聚类引导的注意力调整
        A_group = torch.zeros(N, K, device=x.device)
        for k, group in enumerate(groups):
            if len(group) > 0:
                A_group[:, k] = A[:, group].mean(dim=-1)
        A_group = F.softmax(A_group / self.tau, dim=-1)
        A_prime = A.clone()
        for k, group in enumerate(groups):
            if len(group) > 0:
                A_prime[:, group] *= A_group[:, k:k+1]

        # 单嵌入融合
        f_single = torch.matmul(A_prime, des_tensor)

        # 组级融合
        f_group = torch.zeros(N, dim, device=x.device)
        for k, group in enumerate(groups):
            if len(group) > 0:
                group_weights = A_prime[:, group] / (A_prime[:, group].sum(dim=-1, keepdim=True) + 1e-8)
                f_group_k = torch.matmul(group_weights, des_tensor[group])
                f_group += A_group[:, k:k+1] * f_group_k

        # 多尺度组合
        alpha = self.mlp(torch.cat([x, f_single], dim=-1))
        f = alpha * f_single + (1 - alpha) * f_group

        # 残差连接与归一化
        f_final = x + self.mlp_fusion(f)
        f_output = self.layer_norm(f_final)

        return f_output