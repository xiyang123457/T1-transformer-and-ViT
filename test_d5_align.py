import torch                                       # 张量运算
import torch.nn as nn                              # 用官方的 nn.MultiheadAttention
from attention import MultiHeadAttention           # 我们自己写的多头 (在 attention.py 里)

# 固定随机种子: 让每次运行的输入一致, 结果可复现
torch.manual_seed(0)

B, N, d_model, h = 2, 5, 16, 4                     # batch=2, 序列长=5, 模型维=16, 头数=4

# 输入: (B, N, d_model)
x = torch.randn(B, N, d_model)

# 我们的实现; .eval() 关掉 dropout, 保证对齐时是确定性的
ours = MultiHeadAttention(d_model, h).eval()

# 官方实现; batch_first=False 是官方默认 (输入要求 (N,B,E)), 显式写出来便于理解
# 走 slow path (数学实现), 与我们同源, 最适合比对
ref = nn.MultiheadAttention(d_model, h, batch_first=False).eval()

# ── 共享权重: 只有权重完全一样, 输出才可能一致 ──
# 官方把 Q/K/V 三段投影叠成一个矩阵 in_proj_weight, 形状 (3E, E), 顺序是 [W_q; W_k; W_v]
# 所以按 Q,K,V 顺序 cat 我们的三个 Linear 权重再填进去
with torch.no_grad():                              # 改权重不算梯度, 用 no_grad 包起来更规范
    ref.in_proj_weight.copy_(torch.cat([
        ours.W_q.weight,                           # (E, E)
        ours.W_k.weight,                           # (E, E)
        ours.W_v.weight,                           # (E, E)
    ], dim=0))                                     # 沿第 0 维拼接 -> (3E, E)
    ref.in_proj_bias.copy_(torch.cat([             # bias 同理, 拼成 (3E,)
        ours.W_q.bias,
        ours.W_k.bias,
        ours.W_v.bias,
    ], dim=0))
    ref.out_proj.weight.copy_(ours.out_proj.weight)    # 输出投影 weight (E, E)
    ref.out_proj.bias.copy_(ours.out_proj.bias)        # 输出投影 bias   (E,)

# ── case 1: 无 mask ──
out_ours = ours(x)                                 # 我们的输出 (B, N, E)

x_perm = x.transpose(0, 1)                         # 官方要 (N, B, E): 交换第 0、1 维
out_ref, _ = ref(x_perm, x_perm, x_perm,           # 自注意力: q=k=v 都是同一个 x
                 need_weights=False)               # 不返回注意力权重, 走标准计算路径
out_ref = out_ref.transpose(0, 1)                  # 官方输出 (N,B,E) -> 转回 (B,N,E)

err1 = (out_ours - out_ref).abs().max().item()     # 逐个元素作差, 取最大绝对值 = 最大误差

# ── case 2: causal mask ──
# 上三角 (j>i) 为 True = 屏蔽未来; 官方 attn_mask 语义与我们一致 (True=屏蔽)
causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
out_ours_c = ours(x, mask=causal)
out_ref_c, _ = ref(x_perm, x_perm, x_perm,
                   attn_mask=causal,               # 传入同一张 bool 屏蔽表
                   need_weights=False)
out_ref_c = out_ref_c.transpose(0, 1)
err2 = (out_ours_c - out_ref_c).abs().max().item()

print(f"no-mask  max abs err = {err1:.3e}")        # 期望 < 1e-4
print(f"causal   max abs err = {err2:.3e}")        # 期望 < 1e-4
