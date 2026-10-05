# W1 复盘小结（手写 attention 起步）

> 周期：2026.09.23 – 10.05 ｜ 目标：把 SDPA / 多头自己手写出来，并与官方实现数值对齐

## 一、本周目标达成

| 天 | 任务 | 产出 | 验收证据 |
|---|---|---|---|
| D1 | 建仓 + attention 数学卡 | `notes/attention_walkthrough` | 能口述 QKV / 缩放 / 多头子空间 |
| D2 | 手写 `scaled_dot_product_attention` | `attention.py` | 输出 shape `(B, N, d_k)` 正确 |
| D3 | dropout + causal mask，对齐 `F.scaled_dot_product_attention` | `attention.py` | max abs err = 0 |
| D4 | 手写 `MultiHeadAttention` | `attention.py` | 输出 `(B, N, d_model)` |
| D5 | 与 `nn.MultiheadAttention` 数值对齐 | `test_d5_align.py` | **max abs err = 5.96e-08 < 1e-4** |

## 二、交付物

- `attention.py`
  - `scaled_dot_product_attention(Q, K, V, mask=None, dropout=0.0, training=True)`
  - `class MultiHeadAttention(nn.Module)`
- `test_d5_align.py`：与官方 `nn.MultiheadAttention` 共享权重的对齐脚本
- `notes/attention_walkthrough.{md,html}`：SDPA 逐行讲解

## 三、核心概念（已掌握）

1. **QKV**：Q=找什么、K=能提供什么、V=实际给什么；三者用独立投影学习不同变换。
2. **`1/sqrt(d_k)` 缩放**：维度高 → 点积方差大 → softmax 饱和 → 梯度消失；缩放把方差拉回 O(1)。
3. **`softmax(dim=-1)`**：对每个 query 在所有 key 上归一化，使每行权重和 = 1。
4. **多头 = 多个子空间**：拆 `d_model = h × head_dim`，每个 head 在不同子空间学不同关系，最后拼回。
5. **mask**：一张 bool "禁止表"，True=屏蔽；softmax 前 `masked_fill(-inf)` → 权重归零。

## 四、本周真实踩过的坑（重点复盘）

1. **dropout 写了等于没写**：把丢过的结果赋给 `atten`（少个 `t`），但计算用的还是原 `attn` → dropout 完全失效。
   - 教训：改局部变量后要确认**后续真正用的是它**。
2. **循环导入**：类放进 `attention.py` 后没删 `from attention import ...` → `partially initialized module`。
   - 教训：**同文件内直接用函数名，不要自己 import 自己**。
3. **拼写笔误**：SDPA 形参 `trainning`、属性 `dropot_p`。能跑但会埋雷（用关键字传参会 `TypeError`）。
4. **`test.py` 忘了 import 类** → `NameError`。
5. **官方 `nn.MultiheadAttention` 的格式坑**：默认 `batch_first=False`（输入 `(N,B,E)`）、`need_weights=False`、fast-path 触发条件。

## 五、D5 对齐要点（复用价值高）

- 官方把 Q/K/V 叠成 `in_proj_weight`，形状 `(3E, E)`，顺序 `[W_q; W_k; W_v]`；`in_proj_bias` 同理。
- 对齐流程：共享权重 → 输入 permute 成 `(N,B,E)` → `need_weights=False` → 输出 permute 回 `(B,N,E)` → 比 max abs err。
- `batch_first=False` 走 slow path（数学实现），与我们同源，最适合比对。

## 六、下一步：W2

- **pre-LN vs post-LN**（`LayerNorm` 放子层前还是后）
- **残差连接** `x + sublayer(x)`
- **前馈层 FFN**：`Linear → GELU → Linear`
- 拼成可堆叠的 **Transformer Encoder Block**
