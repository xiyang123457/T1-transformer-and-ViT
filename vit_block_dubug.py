"""
手写 ViTEncoderBlock vs timm blocks[0] 数值对齐 (W2 收口)
要求: err < 1e-4
"""
import timm
import torch
from vit import ViTEncoderBlock

torch.manual_seed(0)                                    # 固定种子 (pretrained=False 权重随机)

# ── 1. timm 模型与目标 block ──
# pretrained=False: 对齐只需两边共用同一套权重, 无需下载预训练
model = timm.create_model('vit_small_patch16_224', pretrained=False)
model.eval()                                            # eval: 关 dropout / drop_path
t_block = model.blocks[0]
t_block.attn.fused_attn = False                         # 关 fused, 与手写 SDPA 同路径

# ── 2. 手写 block ──
my_block = ViTEncoderBlock(d_model=384, num_heads=6, mlp_ratio=4.0,
                           dropout=0.0, init_values=1.0)
my_block.eval()

# ── 3. 权重映射: 把 timm 权重拷进我们的 block ──
with torch.no_grad():
    # LayerNorm: 结构相同, 整体加载
    my_block.norm1.load_state_dict(t_block.norm1.state_dict())
    my_block.norm2.load_state_dict(t_block.norm2.state_dict())

    # qkv: 官方是合并 Linear(384->1152); chunk(3) 拆成 Q/K/V (顺序 q|k|v)
    Wq, Wk, Wv = t_block.attn.qkv.weight.chunk(3, dim=0)
    bq, bk, bv = t_block.attn.qkv.bias.chunk(3, dim=0)
    my_block.attn.W_q.weight.copy_(Wq); my_block.attn.W_q.bias.copy_(bq)
    my_block.attn.W_k.weight.copy_(Wk); my_block.attn.W_k.bias.copy_(bk)
    my_block.attn.W_v.weight.copy_(Wv); my_block.attn.W_v.bias.copy_(bv)

    # 输出投影: 官方 attn.proj -> 我们 attn.out_proj
    my_block.attn.out_proj.weight.copy_(t_block.attn.proj.weight)
    my_block.attn.out_proj.bias.copy_(t_block.attn.proj.bias)

    # MLP: 名字结构一致
    my_block.mlp.fc1.weight.copy_(t_block.mlp.fc1.weight)
    my_block.mlp.fc1.bias.copy_(t_block.mlp.fc1.bias)
    my_block.mlp.fc2.weight.copy_(t_block.mlp.fc2.weight)
    my_block.mlp.fc2.bias.copy_(t_block.mlp.fc2.bias)

    # ls1/ls2: 官方是 Identity(无参数); 我们 gamma 初值 1.0 -> 恒等, 无需拷

# ── 4. 对比 ──
x = torch.randn(2, 197, 384)                            # block 输入即 token 序列 (B,N,D)
with torch.no_grad():
    y_my = my_block(x)
    y_t = t_block(x)
err = (y_my - y_t).abs().max().item()
print(f"ViT block max abs err = {err:.3e}")
print("pass (<1e-4):", err < 1e-4)
