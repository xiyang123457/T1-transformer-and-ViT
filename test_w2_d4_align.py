import timm                        # 官方 ViT
import torch                       # 张量
import torch.nn as nn              # 备用
from vit import ViTEncoderBlock    # 我们手写的 block


torch.manual_seed(0)               # 固定种子: pretrained=False 时权重随机, 固定便于复现

# ── 1. 构建 timm 模型, 取第 0 个 block ──
# pretrained=False: 只搭结构、不下载 (对齐靠"两边共用同一套权重", 无需预训练)
model = timm.create_model('vit_small_patch16_224', pretrained=False)
model.eval()                       # eval: 关 dropout / drop_path
t_block = model.blocks[0]
t_block.attn.fused_attn = False    # 关掉 fused attention, 走与手写 SDPA 相同的数学路径

# ── 2. 构建我们手写的 block ──
my_block = ViTEncoderBlock(d_model=384, num_heads=6, mlp_ratio=4.0,
                           dropout=0.0, init_values=1.0)
my_block.eval()

# ── 3. 权重映射: 把 timm 的权重拷进我们的 block ──
with torch.no_grad():
    # LayerNorm: 结构相同, 整体加载
    my_block.norm1.load_state_dict(t_block.norm1.state_dict())
    my_block.norm2.load_state_dict(t_block.norm2.state_dict())

    # qkv: timm 是合并的 Linear(384->1152); 按行 chunk(3) 拆成 Q/K/V
    #   timm forward 把输出 reshape 成 (B,N,3,heads,head_dim),
    #   所以第 0 块=Q, 第 1 块=K, 第 2 块=V
    Wq, Wk, Wv = t_block.attn.qkv.weight.chunk(3, dim=0)
    bq, bk, bv = t_block.attn.qkv.bias.chunk(3, dim=0)
    my_block.attn.W_q.weight.copy_(Wq); my_block.attn.W_q.bias.copy_(bq)
    my_block.attn.W_k.weight.copy_(Wk); my_block.attn.W_k.bias.copy_(bk)
    my_block.attn.W_v.weight.copy_(Wv); my_block.attn.W_v.bias.copy_(bv)

    # 输出投影: timm 叫 attn.proj, 我们叫 attn.out_proj
    my_block.attn.out_proj.weight.copy_(t_block.attn.proj.weight)
    my_block.attn.out_proj.bias.copy_(t_block.attn.proj.bias)

    # MLP: 名字结构一致, 直接拷
    my_block.mlp.fc1.weight.copy_(t_block.mlp.fc1.weight)
    my_block.mlp.fc1.bias.copy_(t_block.mlp.fc1.bias)
    my_block.mlp.fc2.weight.copy_(t_block.mlp.fc2.weight)
    my_block.mlp.fc2.bias.copy_(t_block.mlp.fc2.bias)

    # ls1/ls2: timm 是 Identity(无参数); 我们的 gamma 初值=1.0 -> 等价恒等, 无需拷

# ── 4. 同输入对比 ──
x = torch.randn(2, 197, 384)       # block 输入就是 token 序列 (B, N, D)
with torch.no_grad():
    y_my = my_block(x)             # 我们
    y_t = t_block(x)               # timm

err = (y_my - y_t).abs().max().item()
print(f"block max abs err = {err:.3e}")   # 期望 < 1e-4
