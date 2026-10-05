import math                        # 数学函数: log() 等
import torch                       # 张量运算
import torch.nn as nn              # nn.Module / nn.Parameter / 各层
import torch.nn.functional as F    # 预留 (本文件暂未使用)
from attention import MultiHeadAttention   # 复用 W1 手写的多头注意力


def positional_encoding(max_len, d_model):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)

    公式 (pos=位置, i=维度对索引 0..d_model/2-1):
        PE[pos, 2i]   = sin( pos / 10000^(2i/d_model) )
        PE[pos, 2i+1] = cos( pos / 10000^(2i/d_model) )

    参数:
        max_len (int): 最大序列长度 -> 结果行数
        d_model (int): 编码维度 (偶数, sin/cos 成对) -> 结果列数
    返回:
        PE (Tensor): shape (max_len, d_model)
    """
    # torch.arange(max_len): 生成 [0,1,...,max_len-1]
    #   dtype=torch.float32: 指定浮点型 (默认 int64), 因为要与 div_term(浮点) 相乘
    # .unsqueeze(1): 在位置 1 插入长度 1 的维度 -> (max_len,) 变 (max_len,1) 列向量
    #   为什么: 便于与 (d_model/2,) 广播相乘, 得到 (max_len, d_model/2)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)     # (max_len, 1)

    # div_term[i] = 10000^(-2i/d_model) = exp( (-2i/d) * ln(10000) )
    #   torch.arange(0, d_model, 2): 步长2 -> 偶数下标 2i = [0,2,4,...,d_model-2]
    #   用 exp/log 代替直接大数幂 10000**(2i/d), 避免精度损失
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))            # (d_model/2,)

    # 容器: (max_len, d_model)
    #   保持 2 维: (max_len, d_model) 与 (B, N, d_model) 相加时按最右维广播即可,
    #   不需要多余的中间维度 (多一维反而会让最左维 max_len 与 B 对不上而报错)
    PE = torch.zeros(max_len, d_model)                                # (max_len, d_model)

    # pos * div_term: (max_len,1) * (d_model/2,) 广播 -> (max_len, d_model/2)
    # PE[:, 0::2]: 所有行、偶数列 -> 填 sin
    # PE[:, 1::2]: 所有行、奇数列 -> 填 cos
    PE[:, 0::2] = torch.sin(pos * div_term)
    PE[:, 1::2] = torch.cos(pos * div_term)

    return PE                                                         # (max_len, d_model)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码 (ViT 实际采用的方案)"""

    def __init__(self, max_len, d_model):
        # super().__init__(): 调父类 nn.Module 初始化, 建立"参数自动登记"机制 (必须)
        super().__init__()

        # nn.Parameter(t): 把张量标记为可训练参数 (会被优化器收集)
        # torch.randn(1, max_len, d_model): 随机初始化位置表; 第 0 维=1 便于对 batch 广播
        # * 0.02: 缩小初值尺度 (与 ViT 官方一致), 避免一开始就盖过输入信号
        self.pose_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x: (B, N, d_model)
        # pose_embed[:, :N, :]: 取前 N 个位置 -> (1, N, d_model); 与 x 相加按第 0 维广播
        return x + self.pose_embed[:, :x.shape[1], :]


class MLP(nn.Module):
    """
    位置无关前馈层 (Position-wise Feed-Forward)
    结构: Linear(d_model -> hidden) -> GELU -> Linear(hidden -> d_model), 逐 token 独立

    参数:
        d_model (int)    : 输入/输出维度
        mlp_ratio (float): 隐藏维倍数, hidden = int(d_model * mlp_ratio), ViT 常规 4.0
        dropout (float)  : dropout 概率 (对齐官方时为 0.0)
    返回:
        (B, N, d_model)
    """

    def __init__(self, d_model, mlp_ratio=4.0, dropout=0.0):
        # super().__init__(): 调父类初始化, 建立参数自动登记机制 (必须)
        super().__init__()
        # int(...): nn.Linear 的维度必须是整数, 显式取整
        hidden = int(d_model * mlp_ratio)         # 隐藏维 (提高维度), 如 384*4 = 1536
        self.fc1 = nn.Linear(d_model, hidden)     # 升维: d_model -> hidden
        self.act = nn.GELU()                      # 激活: 逐元素, ViT 采用 GELU
        self.fc2 = nn.Linear(hidden, d_model)     # 降维: hidden -> d_model
        self.dropout = nn.Dropout(dropout)        # 训练时随机置零, eval() 自动直通; p=0 恒等

    def forward(self, x):
        # x: (B, N, d_model)
        x = self.fc1(x)         # (B, N, hidden)
        x = self.act(x)         # 逐元素 GELU, shape 不变
        x = self.dropout(x)
        x = self.fc2(x)         # (B, N, d_model)
        x = self.dropout(x)
        return x                # (B, N, d_model)


class LayerScale(nn.Module):
    """
    逐通道缩放 (对应 timm ViT block 的 ls1 / ls2)
    公式: y = gamma * x,  gamma 为可学习的逐通道系数
    init_values=1.0 时等价于恒等 (乘 1)

    参数:
        dim (int)          : 通道数 (= d_model)
        init_values (float): gamma 初值
    返回:
        与输入同 shape
    """

    def __init__(self, dim, init_values=1.0):
        super().__init__()
        # nn.Parameter(t): 标记为可训练参数 (会被 model.parameters() 收集)
        # torch.ones((dim)): 全 1 向量; 外层括号只是分组, 等价于 torch.ones(dim)
        # * init_values 设定初值
        # 形状 (dim,): 与 (B,N,dim) 相乘时按最右维广播 (广播口诀: 1 放最左或直接省略)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        # x: (B, N, d_model)
        return self.gamma * x      # (dim,) 广播乘 (B,N,dim) -> (B,N,d_model)


class ViTEncoderBlock(nn.Module):
    """
    ViT Encoder Block (pre-LN)
    公式:
        x = x + ls1( Attn( LN(x) ) )
        x = x + ls2( MLP ( LN(x) ) )
    注意: LN 在子层之前 (pre-LN); 写反成 post-LN(LN(x+Sublayer(x))) 会与 timm 差 0.1 量级

    参数:
        d_model (int)      : token 维度
        num_heads (int)    : 注意力头数
        mlp_ratio (float)  : MLP 隐藏维倍数
        dropout (float)    : dropout 概率 (对齐时 0.0)
        init_values (float): LayerScale 初值
    返回:
        (B, N, d_model)
    """

    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0, init_values=1.0):
        super().__init__()

        # ── 子层 1: 多头注意力 ──
        self.norm1 = nn.LayerNorm(d_model)                       # LN 放在 Attn 之前 (pre-LN 关键)
        # 复用 W1 的 MultiHeadAttention; 第 3 个位置参数即 dropout
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ls1 = LayerScale(d_model, init_values)              # 对应 timm 的 ls1

        # ── 子层 2: MLP ──
        self.norm2 = nn.LayerNorm(d_model)                       # LN 放在 MLP 之前
        self.mlp = MLP(d_model, mlp_ratio, dropout)
        self.ls2 = LayerScale(d_model, init_values)              # 对应 timm 的 ls2

    def forward(self, x):
        # x: (B, N, d_model)
        # 子层1: LN -> Attn -> ls1 -> 残差相加 (顺序不能乱)
        x = x + self.ls1(self.attn(self.norm1(x)))               # (B, N, d_model)
        # 子层2: LN -> MLP -> ls2 -> 残差相加
        x = x + self.ls2(self.mlp(self.norm2(x)))                # (B, N, d_model)
        return x                                                 # (B, N, d_model)
