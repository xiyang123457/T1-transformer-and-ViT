import torch                       # 张量
import torch.nn as nn              # 层

# ── 基本参数 (ViT-S/16) ──
B = 2                              # 批大小
C = 3                              # 输入通道 (RGB)
H = W = 224                        # 图像高宽
P = 16                             # patch 边长
d_model = 384                      # token 维度 (ViT-S)

x = torch.randn(B, C, H, W)        # 假输入图像 (B,3,224,224)
print("image       :", tuple(x.shape))

# ── 算 patch 数量 ──
# 224 / 16 = 14, 网格 14x14
assert H % P == 0 and W % P == 0   # 高宽必须能被 patch 整除
grid = H // P                      # 14
N = grid * grid                    # 14*14 = 196
print(f"grid        : {grid}x{grid} = {N} patches")

# ── patch 投影 ──
# nn.Conv2d(3, 384, kernel_size=P, stride=P): 用 kernel/stride 都等于 patch 边长的卷积,
#   等价于"把图切成不重叠 patch + 每个 patch 线性投影" (timm 的 patch_embed 就这么实现)
#   参数意义: in_channels=3, out_channels=d_model=384, kernel_size=16, stride=16
patch_embed = nn.Conv2d(C, d_model, kernel_size=P, stride=P)
tok = patch_embed(x)               # (B, 384, 14, 14)
print("patch_embed :", tuple(tok.shape))

# flatten(2): 把第 2 维起的所有维展平 -> (B, 384, 14*14) = (B, 384, 196)
tok = tok.flatten(2)               # (B, 384, 196)
# transpose(1,2): 交换第1、2维 -> (B, 196, 384) = (B, N, d_model)
tok = tok.transpose(1, 2)          # (B, 196, 384)
print("tokens      :", tuple(tok.shape))

# ── 拼 [CLS] ──
# 可学习的 CLS 向量, 形状 (1,1,d_model); expand 到 batch: (B,1,d_model)
cls = nn.Parameter(torch.zeros(1, 1, d_model)).expand(B, -1, -1)   # (B,1,384)
# 沿序列维(dim=1)拼接: (B,1,384) + (B,196,384) -> (B,197,384)
tok = torch.cat([cls, tok], dim=1)                                 # (B, 197, 384)
print("with CLS    :", tuple(tok.shape))

# ── 加位置编码 ──
# pos_embed (1,197,384) 与 tok (B,197,384) 相加, 按第0维广播
pos_embed = nn.Parameter(torch.zeros(1, 197, d_model))             # (1, 197, 384)
tok = tok + pos_embed                                              # (B, 197, 384)
print("with pos    :", tuple(tok.shape))
