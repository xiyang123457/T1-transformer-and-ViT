import torch
from vit import ViTEncoderBlock

blk = ViTEncoderBlock(d_model=384, num_heads=6)
x = torch.randn(2, 197, 384)
y = blk(x)
print(y.shape)          # 期望 torch.Size([2, 197, 384])
print(y.isnan().any())  # 期望 False
