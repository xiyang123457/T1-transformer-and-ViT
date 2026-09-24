import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q,K,V,mask=None):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)

    参数:
        Q: Query,  shape (B, N, d_k)  -- B=批大小, N=序列长度, d_k=每个头的维度
        K: Key,    shape (B, N, d_k)
        V: Value,  shape (B, N, d_k)
        mask: D2 暂不处理, 留到 D3 加 causal / padding mask
    返回:
        out: shape (B, N, d_k)
    """
    # 0. 取 d_k 用于缩放: Q 的最后一个维度就是 d_k
    d_k = Q.size(-1)

    # 1. 算分数: scores = Q @ K^T
    #    K.transpose(-2, -1) 把 (B, N, d_k) 的最后两维交换 -> (B, d_k, N)
    #    (B, N, d_k) @ (B, d_k, N) -> (B, N, N)
    #    含义: 每个 query(共 N 个) 对每一个 key(共 N 个) 打一个分
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. 缩放: 除以 sqrt(d_k)
    #    维度越高, 点积方差越大 -> 数值冲进 softmax 饱和区 -> 梯度消失
    #    除以 sqrt(d_k) 把分数方差拉回 O(1)
    scores = scores / math.sqrt(d_k)

    # 3. 归一化: 对每个 query 的 N 个分数做 softmax
    #    dim=-1 作用在最后一维(那 N 个 key 上), 使每行和为 1 -> 注意力权重
    attn = F.softmax(scores, dim=-1)

    # 4. 取内容: 用注意力权重对 V 加权求和
    #    (B, N, N) @ (B, N, d_k) -> (B, N, d_k)
    #    含义: 按注意力权重把各 value 混合成每个位置的新表示
    out = torch.matmul(attn, V)

    return out
    
    