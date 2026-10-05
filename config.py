"""
W3-D3 — 实验配置: 把"受控实验"的全部变量钉死在一个 dataclass

用途:
    W4 对照矩阵的每个格子 (架构 × 策略 × 数据量 × 种子) 都由一个 ProtocolCfg 描述;
    只改这一个对象, 数据/模型/lr/评估全流程同步变 -> 保证"一次只改一个变量"。
输入: 无 (纯配置)
输出: 一个可传入 run_protocol(cfg) 的配置对象

跑完应看到的自检数字 (实例化默认 cfg 后):
    - cfg.num_classes == 102
    - cfg.mean == (0.485, 0.456, 0.406) 且 cfg.std == (0.229, 0.224, 0.225)  (须与 data.py 一致)
    - cfg.ratio 属于 {0.1, 0.25, 0.5, 1.0}
"""
# ==== 1. 依赖导入 ====
from dataclasses import dataclass      # @dataclass 自动生成 __init__ / __repr__

# ==== 2. 配置对象 (全部给默认值 = 一个能直接跑的基线格子) ====
@dataclass
class ProtocolCfg:
    """一次协议运行的全部配置 (W4 矩阵的一个格子)"""

    arch: str = "resnet50"
    # 变量 arch: str, 模型架构标识
    #   示例值: "resnet50" / "vit_small_patch16_224" / "deit_tiny_patch16_224"
    #   为什么: 字符串直接喂 timm / torchvision 建模型; 三架构共用同一套协议代码

    mode: str = "Z2"
    # 变量 mode: str, 迁移档位
    #   示例值: "Z0"(冻结+NCM) / "Z1"(只训头) / "Z2"(全参微调)
    #   为什么: run_protocol 按它分派; 三档必须同一套数据/评估口径才可比

    ratio: float = 1.0
    # 变量 ratio: float, train 子集比例
    #   示例值: 0.1 / 0.25 / 0.5 / 1.0
    #   为什么: 与 W3-D1 的嵌套子集对齐, "数据量"维只改这一个数, 其余不动

    seed: int = 42
    # 变量 seed: int, 随机种子
    #   示例值: 42
    #   为什么: 全流程固定它, 结果才可复现; 复现是"受控实验"的底线

    epochs: int = 30
    # 变量 epochs: int, 最大训练轮数 (上界, 由早停决定实际停止)
    #   示例值: 30
    #   为什么: 子集小的时候几轮就收敛, 设大以免早停前撞上限

    batch: int = 32
    # 变量 batch: int, 批大小
    #   示例值: 32 (ResNet-50 在 8GB 显存下的经验值; ViT-S 建议 16)
    #   为什么: 显存不够就降; 太小会让 BN 统计不稳、训练抖动

    lr_head: float = 1e-3
    # 变量 lr_head: float, 分类头学习率 (大)
    #   示例值: 1e-3
    #   为什么: 头是随机初始化的, 需要快学; Z1 只用这一个 lr

    lr_backbone: float = 1e-4
    # 变量 lr_backbone: float, 骨干学习率 (小, 仅 Z2 用)
    #   示例值: 1e-4
    #   为什么: 骨干已预训练, 大 lr 会灾难性遗忘预训练特征

    weight_decay: float = 1e-4
    # 变量 weight_decay: float, AdamW 权重衰减
    #   示例值: 1e-4
    #   为什么: 微调常用值, 抑制过拟合

    patience: int = 5
    # 变量 patience: int, 早停耐心
    #   示例值: 5
    #   为什么: val 连续 5 轮不升则停; 小则误停, 大则浪费算力

    num_workers: int = 0
    # 变量 num_workers: int, DataLoader 子进程数
    #   示例值: 0 (Windows 先 0)
    #   为什么: Windows 多进程易 spawn 报错, 跑通后再试 2/4

    num_classes: int = 102
    # 变量 num_classes: int, 类别数
    #   示例值: 102 (Flowers102)
    #   为什么: 写进 config 而不是散落各处, 换数据集只改一处

    input_size: int = 224
    # 变量 input_size: int, 输入边长
    #   示例值: 224
    #   为什么: 与三个 backbone 的预训练分辨率一致, 不改

    mean: tuple = (0.485, 0.456, 0.406)
    # 变量 mean: tuple[float,float,float], 归一化均值 (R,G,B)
    #   示例值: (0.485, 0.456, 0.406)
    #   为什么: ImageNet-1k 统计量; 三架构统一才可比

    std: tuple = (0.229, 0.224, 0.225)
    # 变量 std: tuple[float,float,float], 归一化标准差
    #   示例值: (0.229, 0.224, 0.225)
    #   为什么: 与改代码里的 data.py 必须一致, 否则推理分布错、精度虚降

# ==== 怎么验证跑对了 ====
# 1) `from config import ProtocolCfg; c=ProtocolCfg()` 不报错;
# 2) `c.num_classes == 102`, `c.mean == (0.485, 0.456, 0.406)`;
# 3) 打印 c 能看到全部字段 (dataclass 自动生成 __repr__)。
