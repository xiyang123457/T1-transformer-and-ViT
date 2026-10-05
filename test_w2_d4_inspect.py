import timm                        # 加载官方 ViT
import torch                       # 张量
import torch.nn as nn              # 判断模块类型用

# timm.create_model('名称', pretrained=True)
#   作用: 按名字构建模型; pretrained=True 下载并加载官方预训练权重 (首次会联网下载)
model = timm.create_model('vit_small_patch16_224', pretrained=True)
model.eval()                       # eval(): 关掉 dropout / drop_path, 保证后续对齐确定性

block = model.blocks[0]            # 取第 0 个 encoder block (blocks 是 ModuleList)
print(block)                       # 打印整体结构: 显示每个子层名与类型
print("=" * 70)

# named_parameters(): 遍历 "参数名 -> 参数", 看真实字段名和形状
for name, p in block.named_parameters():
    print(f"{name:28s} {tuple(p.shape)}")
print("=" * 70)

# ls1 / ls2 到底是 Identity 还是有参数的 LayerScale?
print("type(ls1)        :", type(block.ls1).__name__)
print("type(ls2)        :", type(block.ls2).__name__)
print("is Identity(ls1) :", isinstance(block.ls1, nn.Identity))
print("is Identity(ls2) :", isinstance(block.ls2, nn.Identity))
print("=" * 70)

# 注意力的 qkv 是不是"合并"的一个 Linear
print("type(attn)   :", type(block.attn).__name__)
print("qkv.weight   :", tuple(block.attn.qkv.weight.shape))    # 期望 (1152, 384) = 3*384
print("proj.weight  :", tuple(block.attn.proj.weight.shape))   # 期望 (384, 384)
print("fused_attn   :", getattr(block.attn, 'fused_attn', 'N/A'))
