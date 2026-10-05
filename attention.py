import torch                                          # 张量运算 (matmul / softmax 等)
import torch.nn.functional as F                       # 函数式 API: softmax, dropout
import torch.nn as nn                                 # 神经网络层与模型基类
import math                                           # 用于 sqrt(d_k)


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=0.0, training=True):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)

    参数:
        Q, K, V : shape (B, N, d_k)  -- B=批大小, N=序列长度, d_k=每个头的维度
        mask    : bool 张量, True=屏蔽; None 表示不屏蔽
        dropout : dropout 概率, 对齐官方时必须设 0.0
        training: 是否训练态; False 时 F.dropout 直通不丢 -> 实现 "eval 不 drop"
    返回:
        out : shape (B, N, d_k)
    """
    # 0. 取 d_k 用于缩放: Q 的最后一维就是 d_k
    d_k = Q.size(-1)

    # 1. 算分数: scores = Q @ K^T
    #    K.transpose(-2, -1): 交换最后两维 (..., N, d_k) -> (..., d_k, N)
    #    (B, N, d_k) @ (B, d_k, N) -> (B, N, N)
    #    含义: 每个 query (共 N 个) 对每个 key (共 N 个) 打一个分
    scores = torch.matmul(Q, K.transpose(-2, -1))          # (B, N, N)

    # 2. 缩放: 除以 sqrt(d_k)
    #    维度越高点积方差越大 -> 数值冲进 softmax 饱和区 -> 梯度消失
    #    除以 sqrt(d_k) 把分数方差拉回 O(1)
    scores = scores / math.sqrt(d_k)                       # (B, N, N)

    # 3. 施加 mask (位置: 缩放之后、softmax 之前)
    #    masked_fill(mask, -inf): 被屏蔽处分数变 -inf, softmax 后权重 -> 0
    #    注意语义: mask 为 True 表示 "屏蔽"
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 4. 归一化: 对每个 query 的 N 个分数做 softmax
    #    dim=-1 作用在最后一维 (那 N 个 key 上), 使每行权重和为 1
    attn = F.softmax(scores, dim=-1)                       # (B, N, N)

    # 5. dropout (位置: softmax 之后、加权之前, 标准位置)
    #    training=False 时 F.dropout 原样返回 -> eval 不丢
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout, training=training)

    # 6. 取内容: 用注意力权重对 V 加权求和
    #    (B, N, N) @ (B, N, d_k) -> (B, N, d_k)
    out = torch.matmul(attn, V)                            # (B, N, d_k)

    return out


class MultiHeadAttention(nn.Module):
    """
    多头自注意力 (Multi-Head Self-Attention)
    数据流: 投影(Q/K/V) -> 拆头 -> 各 head 独立 SDPA -> 拼头 -> 输出投影
    输入 / 输出 shape 均为 (B, N, d_model)
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        # 继承 nn.Module 必须调父类初始化, 才能启用 "参数自动登记" 等机制
        super().__init__()

        # 保险丝: d_model 必须能被 num_heads 整除, 否则拆不出等长子空间
        assert d_model % num_heads == 0

        # 存为对象属性, 供 forward / _split_heads 使用
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads        # 整除; 每个 head 的维度, 即 d_k
        self.dropout_p = dropout                    # 暂存, 前向时传给 SDPA

        # 三个独立线性层: Q/K/V 要学不同变换, 参数不共享
        # nn.Linear(d_model, d_model): 同维度线性变换, 计算 y = x @ W.T + b
        self.W_q = nn.Linear(d_model, d_model)      # 生成 Q
        self.W_k = nn.Linear(d_model, d_model)      # 生成 K
        self.W_v = nn.Linear(d_model, d_model)      # 生成 V

        # 输出投影: 把拼回来的多头结果再线性融合一次 (让各 head 信息交互)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        # 内部方法 (下划线是约定); 作用: (B, N, d_model) -> (B, h, N, head_dim)
        B, N, _ = x.shape   # 解包形状; _ 表示占位、不关心

        # 把最后一维 d_model 拆成 (h, head_dim) 两维; 输入连续, 可直接 view
        x = x.view(B, N, self.num_heads, self.head_dim)     # -> (B, N, h, head_dim)

        # 交换第 1、2 维 (0=B, 1=N, 2=h), 把 head 维挪到 batch 后面
        # 换头后每个 head 是独立的 (N, head_dim), 正好是 SDPA 需要的形状
        x = x.transpose(1, 2)                               # -> (B, h, N, head_dim)

        return x

    def forward(self, x, mask=None):
        # forward 是 PyTorch 约定的前向入口: 调用 mha(x) 会自动执行它
        B, N, _ = x.shape   # 记录 B、N, 后面拼头 reshape 时要用

        # ── 1. QKV 投影 ── 各自 (B, N, d_model)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # ── 2. 拆头 ── (B, N, d_model) -> (B, h, N, head_dim)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # ── 3. 各 head 独立做 SDPA ──
        # SDPA 只看最后两维 (N, head_dim), 这里正好是它, B 与 h 被并行当作 "批"
        # training=self.training: train() 时为 True (丢), eval() 时为 False (不丢)
        out = scaled_dot_product_attention(
            q, k, v,
            mask=mask,                  # 透传屏蔽表
            dropout=self.dropout_p,     # >0 才真的丢
            training=self.training,     # nn.Module 自带属性, 自动随 train/eval 切换
        )                               # 输出 (B, h, N, head_dim)

        # ── 4. 拼头 ── (B, h, N, head_dim) -> (B, N, d_model)
        out = out.transpose(1, 2)                   # head 维挪回 seq 后 -> (B, N, h, head_dim)
        out = out.reshape(B, N, self.d_model)       # 合并 (h, head_dim) -> d_model
        # 注意用 reshape 而非 view: 上一步 transpose 后内存不连续, view 会报错

        # ── 5. 输出投影 ── 最终 (B, N, d_model), 与输入同形状, 可喂给下一层
        out = self.out_proj(out)

        return out
