"""
attention 三种实现的误差矩阵 (W1 收口)
对比: 手写 SDPA vs F.sdpa ; 手写 MHA vs nn.MultiheadAttention
场景: 无 mask / causal; 要求全部 < 1e-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import scaled_dot_product_attention, MultiHeadAttention

torch.manual_seed(0)                                   # 固定种子, 结果可复现

B, N, d_model, heads = 2, 5, 16, 4
# 上三角(j>i)为 True = 屏蔽未来; 官方 is_causal 与本 mask 等价
causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

# ── 1. SDPA: 手写 vs 官方 ──
Q = torch.randn(B, N, d_model)
K = torch.randn(B, N, d_model)
V = torch.randn(B, N, d_model)
sdpa_nomask = (scaled_dot_product_attention(Q, K, V)
               - F.scaled_dot_product_attention(Q, K, V)).abs().max().item()
sdpa_causal = (scaled_dot_product_attention(Q, K, V, mask=causal)
               - F.scaled_dot_product_attention(Q, K, V, is_causal=True)).abs().max().item()

# ── 2. MHA: 手写 vs 官方 (共享权重) ──
x = torch.randn(B, N, d_model)
my_mha = MultiHeadAttention(d_model, heads).eval()                       # 我们的
ref_mha = nn.MultiheadAttention(d_model, heads, batch_first=False).eval()  # 官方 (输入 (N,B,E))

with torch.no_grad():
    # 官方把 Q/K/V 叠成 in_proj_weight (3E,E), 顺序 [Wq;Wk;Wv]
    ref_mha.in_proj_weight.copy_(torch.cat(
        [my_mha.W_q.weight, my_mha.W_k.weight, my_mha.W_v.weight], dim=0))
    ref_mha.in_proj_bias.copy_(torch.cat(
        [my_mha.W_q.bias, my_mha.W_k.bias, my_mha.W_v.bias], dim=0))
    ref_mha.out_proj.weight.copy_(my_mha.out_proj.weight)
    ref_mha.out_proj.bias.copy_(my_mha.out_proj.bias)

xp = x.transpose(0, 1)                                  # (N,B,E) 官方默认格式
y_my = my_mha(x)
y_my_c = my_mha(x, mask=causal)
y_ref, _ = ref_mha(xp, xp, xp, need_weights=False)
y_ref_c, _ = ref_mha(xp, xp, xp, attn_mask=causal, need_weights=False)
mha_nomask = (y_my - y_ref.transpose(0, 1)).abs().max().item()
mha_causal = (y_my_c - y_ref_c.transpose(0, 1)).abs().max().item()

# ── 打印误差矩阵 ──
rows = [
    ("手写 SDPA vs F.sdpa", sdpa_nomask, sdpa_causal),
    ("手写 MHA  vs nn.MultiheadAttention", mha_nomask, mha_causal),
]
print(f"{'对比':<38}{'no-mask':>13}{'causal':>13}")
for name, e1, e2 in rows:
    print(f"{name:<38}{e1:>13.3e}{e2:>13.3e}")
worst = max(sdpa_nomask, sdpa_causal, mha_nomask, mha_causal)
print("all < 1e-4:", worst < 1e-4, f"(worst={worst:.3e})")
