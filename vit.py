import math                        # 数学函数: log() 等
import torch                       # 张量运算
import torch.nn as nn              # nn.Module / nn.Parameter
import torch.nn.functional as F    # 预留 (本文件暂未用)


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
