"""
W4-D1 — 实验配置 (唯一配置源, 口径 15): 架构注册表 + 单格配置

用途:
    用「架构注册表 + 单格配置」描述对照矩阵的每一格 (架构 × 策略 × 数据量 × 种子);
    所有模块 (data / protocol / run_matrix) 只读本文件, 不各自硬编码常数。
输入: 无 (纯配置)
输出: `ArchSpec` / `ARCH_REGISTRY` / `ProtocolCfg`

跑完应看到的自检数字 (见文件末尾「怎么验证跑对了」):
    - `len(ARCH_REGISTRY) == 4`，代号 r50 / vits / vits1k / deitt
    - `ProtocolCfg(arch="vits", mode="Z2", data=25, seed=42).eid == "vits-z2-25-s42"`
    - `preproc_str` 对 ViT-S 得 `vit_small/augreg_in21k_ft_in1k/0.5-0.5`
    - `ProtocolCfg(mode="Z0").train_aug is False`；`ProtocolCfg(mode="Z1").train_aug is True`
"""
# ==== 1. 依赖导入 ====
from dataclasses import dataclass   # @dataclass 自动生成 __init__ / __repr__ / __eq__

# ==== 2. 架构注册表 (口径 1/3/4: 型号 + 预训练 tag + 评估 token + 等效 batch) ====
@dataclass(frozen=True)
class ArchSpec:
    """一个架构的全部固定属性 (整周不变, 不随格子变化)"""

    code: str
    # 变量 code: str, 架构代号 (用于 eid 与 CLI)
    #   示例值: "r50" / "vits" / "vits1k" / "deitt"
    #   为什么用短代号: eid 要紧凑可读 (vits-z2-25-s42), 且与 CLI 参数一一对应

    library: str
    # 变量 library: str, 模型来源库
    #   示例值: "torchvision" / "timm"
    #   为什么: 两者建模型与取预处理的方式不同, 用它分派 (ResNet 走 torchvision, ViT/DeiT 走 timm)

    model_id: str
    # 变量 model_id: str, 模型名 (不含 tag)
    #   示例值: "resnet50" / "vit_small_patch16_224" / "deit_tiny_distilled_patch16_224"
    #   为什么: timm 会拼上 tag 后用 create_model 建; torchvision 直接用

    tag: str
    # 变量 tag: str, 预训练权重标签 (口径 4: ViT-S 两个起点要区分)
    #   示例值: "IMAGENET1K_V2" / "augreg_in21k_ft_in1k" / "augreg_in1k" / "fb_in1k"
    #   为什么: 记录进 preproc 列 -> "起点不同"是维度 A 最致命的混杂, 必须可追溯

    short: str
    # 变量 short: str, 记录用短名 (preproc 列第一段)
    #   示例值: "resnet50" / "vit_small" / "deit_tiny_distilled"
    #   为什么: 报告的 preproc 列要人可读, 且与 tag 组合后能唯一确定"起点"

    eval_token: str
    # 变量 eval_token: str, 评估取哪一路特征 (口径 C / A1 枚举)
    #   示例值: "gap" (ResNet, 全局平均池化) / "cls" (ViT / DeiT 的 [CLS])
    #   为什么: DeiT 蒸馏版 eval 下 model(x) 会返回两头平均, 必须显式走 head -> 写死为 cls

    batch: int
    # 变量 batch: int, 该架构的等效 batch (口径 13: 三架构统一)
    #   示例值: 32 (W3 实测: R50 27.1s / ViT-S 22.8s / DeiT 11.8s 每 epoch)
    #   为什么统一 32: 等效 batch 不同 -> lr 口径不同 -> 组间不可比

    is_distilled: bool
    # 变量 is_distilled: bool, 是否为"带第二个蒸馏头"的结构 (口径 1)
    #   示例值: DeiT-Ti 为 True, 其余 False
    #   为什么: W4 只用 head、不用 head_dist; 标出来便于断言"头确实存在且是 102 类"


ARCH_REGISTRY: dict[str, ArchSpec] = {
    # ⚠ 口径 1: DeiT 主型号必须是 distilled 版 —— `deit_tiny_patch16_224` 无 dist_token/head_dist
    "r50": ArchSpec(
        code="r50", library="torchvision", model_id="resnet50",
        tag="IMAGENET1K_V2", short="resnet50", eval_token="gap",
        batch=32, is_distilled=False,
    ),
    "vits": ArchSpec(
        # 口径 4 主表起点: ImageNet-21k 预训练 + 1k 微调 (timm 默认 tag)
        code="vits", library="timm", model_id="vit_small_patch16_224",
        tag="augreg_in21k_ft_in1k", short="vit_small", eval_token="cls",
        batch=32, is_distilled=False,
    ),
    "vits1k": ArchSpec(
        # 口径 4 起点消融: 只有 1k 预训练 (与 DeiT 同为 1k, 才能拆分"预训练数据量"这个混杂)
        code="vits1k", library="timm", model_id="vit_small_patch16_224",
        tag="augreg_in1k", short="vit_small", eval_token="cls",
        batch=32, is_distilled=False,
    ),
    "deitt": ArchSpec(
        code="deitt", library="timm", model_id="deit_tiny_distilled_patch16_224",
        tag="fb_in1k", short="deit_tiny_distilled", eval_token="cls",
        batch=32, is_distilled=True,
    ),
}

# ==== 3. 枚举常量 (口径 16②③: 只在这里定义一次, 其余模块 import) ====
MODES = ("Z0", "Z1", "Z2")
# 变量 MODES: tuple[str, ...], 迁移策略枚举
#   示例值: ("Z0", "Z1", "Z2")
#   为什么: run_matrix 校验 CLI 输入, 防止写出 "z2"/"Z3" 这类脏值
DATA_TIERS = (10, 25, 50, 100)
# 变量 DATA_TIERS: tuple[int, ...], train 子集档位 (百分比, 口径 11)
#   示例值: (10, 25, 50, 100) -> 每类 1/2/5/10 张 -> 102/204/510/1020 张
AUG_CHOICES = ("none", "basic", "strong")
# 变量 AUG_CHOICES: tuple[str, ...], 增强档枚举; W4 全部 "basic" ("strong" 是 W5 维度 D)
EVAL_TOKENS = ("cls", "gap", "cls+dist_avg")
# 变量 EVAL_TOKENS: tuple[str, ...], 评估 token 枚举 (清单 A1)
DEGRADED_CHOICES = ("none", "micro_batch_reduced", "epoch_halved", "res_down")
# 变量 DEGRADED_CHOICES: tuple[str, ...], 降级标记枚举 (口径 12: 只允许整周统一降级)

SUBSET_SEED = 42
# 变量 SUBSET_SEED: int, 四档 train 子集的抽样种子 (与训练种子解耦!)
#   示例值: 42
#   为什么必须独立: 子集一旦定下就是"数据集本身"; W7 补种子时只换训练种子 (43/44),
#   子集必须还是同一批 -> 否则 43/44 号跑的是另一个数据集, 误差棒算出来是假的

# ==== 4. 单格配置 (矩阵的一个格子) ====
@dataclass
class ProtocolCfg:
    """一次协议运行的全部配置 (26 列里身份 10 列的来源)"""

    arch: str = "r50"
    # 变量 arch: str, 架构代号 (ARCH_REGISTRY 的键)
    #   示例值: "r50" / "vits" / "vits1k" / "deitt"
    #   为什么用代号而非型号名: eid 紧凑、CLI 好敲, 型号/tag 由注册表查得

    mode: str = "Z2"
    # 变量 mode: str, 迁移策略 (MODES 之一)
    #   示例值: "Z0"(NCM 零训练) / "Z1"(只训头) / "Z2"(全参微调)
    #   为什么: run_protocol 按它分派; 三档必须同一套数据/评估口径才可比

    data: int = 100
    # 变量 data: int, train 子集百分比 (DATA_TIERS 之一)
    #   示例值: 100 -> 1020 张; 25 -> 204 张
    #   为什么用百分比而非张数: 与 eid/CSV 列一致, 张数由 data.py 按子集文件解析

    seed: int = 42
    # 变量 seed: int, 随机种子
    #   示例值: 42 (W4 全部 42; 补种子 43/44 在 W7)
    #   为什么: 全流程固定它才可复现; D1 自检要求"重跑数字完全一致"

    epochs: int = 30
    # 变量 epochs: int, 最大训练轮数 (上界, 实际由早停决定) —— 口径 14
    #   示例值: 30
    #   为什么: 1020 张 30 轮足够收敛; 设大以免早停前撞上限

    batch: int = 32
    # 变量 batch: int, 等效 batch (口径 13) —— 三架构统一
    #   示例值: 32
    #   为什么: 统一才能保证 lr 口径一致; OOM 时先降 micro-batch 再用梯度累积补回

    lr_head: float = 1e-3
    # 变量 lr_head: float, 分类头学习率
    #   示例值: 1e-3
    #   为什么: 头是随机初始化的; W4 只跑不调参, 这个值是冻结口径

    lr_backbone: float = 1e-4
    # 变量 lr_backbone: float, 骨干学习率 (仅 Z2 用)
    #   示例值: 1e-4
    #   为什么: 骨干已预训练, 大 lr 会灾难性遗忘

    weight_decay: float = 1e-4
    # 变量 weight_decay: float, AdamW 权重衰减; 示例值 1e-4; 抑制过拟合

    patience: int = 5
    # 变量 patience: int, 早停耐心 (口径 14); 示例值 5; val 连续 5 轮不升则停

    num_workers: int = 0
    # 变量 num_workers: int, DataLoader 子进程数
    #   示例值: 0 (Windows 下多进程易 spawn 出错; 长批跑建议 0 或 2)
    #   为什么: 稳定性优先, 数据量小 (最多 1020 张), 加载不是瓶颈

    num_classes: int = 102
    # 变量 num_classes: int, 类别数 (Flowers102); 换数据集只改这一处

    input_size: int = 224
    # 变量 input_size: int, 输入边长; 示例值 224; 与三个 backbone 预训练分辨率一致

    distill: bool = False
    # 变量 distill: bool, 是否启用蒸馏 (口径 A: W4 全为 False)
    #   示例值: False
    #   为什么: 维度 D (有/无 teacher) 是 W5 的事; W4 只训 head, head_dist 不参与 loss

    eval_token: str = ""
    # 变量 eval_token: str, 评估 token (空串 = 用注册表默认)
    #   示例值: "cls" (ViT/DeiT) / "gap" (ResNet)
    #   为什么: 空串让 __post_init__ 自动取 spec.eval_token, 避免手填错

    aug: str = "basic"
    # 变量 aug: str, 增强档 (AUG_CHOICES); W4 全部 "basic"

    degraded: str = "none"
    # 变量 degraded: str, 降级标记 (DEGRADED_CHOICES); 口径 12: 只允许整周统一降级

    note: str = ""
    # 变量 note: str, 一句话结论 (26 列里唯一人手填的列)
    #   示例值: "符合预期" / "异常, 待查"
    #   为什么: 记录"这组数字说明什么", 不填 = 这组作废

    save_detail: bool = False
    # 变量 save_detail: bool, 是否落逐类 perclass.csv + confusion.npy (口径 18)
    #   示例值: False (仅报告主表最终配置组打开)
    #   为什么: W4 不需要, 打开会徒增落盘

    def __post_init__(self):
        # 设计决策: 用 __post_init__ 做"默认值回填 + 脏值拦截", 而不是散落各模块校验
        #   好处: 任何入口 (CLI/测试/notebook) 建 cfg 时都会被同一套规则约束
        if self.arch not in ARCH_REGISTRY:
            raise ValueError(f"未知 arch={self.arch}; 应为 {tuple(ARCH_REGISTRY)}")
        if self.mode not in MODES:
            raise ValueError(f"未知 mode={self.mode}; 应为 {MODES}")
        if self.data not in DATA_TIERS:
            raise ValueError(f"未知 data={self.data}; 应为 {DATA_TIERS}")
        if not self.eval_token:                        # 空串 -> 取该架构注册表默认
            self.eval_token = ARCH_REGISTRY[self.arch].eval_token
        if self.eval_token not in EVAL_TOKENS:
            raise ValueError(f"未知 eval_token={self.eval_token}; 应为 {EVAL_TOKENS}")
        if self.aug not in AUG_CHOICES:
            raise ValueError(f"未知 aug={self.aug}; 应为 {AUG_CHOICES}")
        if self.degraded not in DEGRADED_CHOICES:
            raise ValueError(f"未知 degraded={self.degraded}; 应为 {DEGRADED_CHOICES}")

    # ---- 派生属性 (不占字段, 由上面字段算出) ----
    @property
    def spec(self) -> ArchSpec:
        """取本格对应的架构注册项"""
        return ARCH_REGISTRY[self.arch]

    @property
    def ratio(self) -> float:
        # 变量 ratio: float, 抽样比例 = data / 100 (data.py 的抽样函数要 0~1)
        #   示例值: 0.25 (data=25) / 1.0 (data=100)
        return self.data / 100.0

    @property
    def train_aug(self) -> bool:
        # 变量 train_aug: bool, 训练集是否用随机增强 (口径 6 / B)
        #   示例值: Z0 -> False (只提特征); Z1/Z2 -> True
        #   为什么: 与 P1 同口径 —— P1 的 head-only 与 full-FT 共用同一套在线增强
        return self.mode != "Z0"

    @property
    def eid(self) -> str:
        # 变量 eid: str, 实验编号 (清单 A1)
        #   示例值: "vits-z2-25-s42"
        #   为什么: 唯一编号; 文件名、CSV 主键、日志都靠它
        return f"{self.arch}-{self.mode.lower()}-{self.data}-s{self.seed}"

    def preproc_str(self, mean, std) -> str:
        """
        生成 `preproc` 记录列 (清单 A1 / 口径 3 的落盘载体)

        参数:
            mean (tuple[float,...]): 该架构实际使用的归一化均值
            std  (tuple[float,...]): 该架构实际使用的归一化标准差
        返回:
            str, 例 "vit_small/augreg_in21k_ft_in1k/0.5-0.5" 或
                 "resnet50/IMAGENET1K_V2/0.485,0.456,0.406-0.229,0.224,0.225"
        """
        # 设计决策: 这里接收"实际用到的" mean/std, 而不是自己去查注册表
        #   原因: 记录的目的是"事后能证明当时喂了什么分布", 必须来自真实构建的 transform
        return f"{self.spec.short}/{self.spec.tag}/{_fmt_vec(mean)}-{_fmt_vec(std)}"

    def identity(self) -> dict:
        """身份 10 列 (供 run_matrix 写 JSON / CSV; preproc 由调用方补上)"""
        return {
            "eid": self.eid,
            "arch": self.arch,
            "mode": self.mode.upper(),
            "data": self.data,
            "aug": self.aug,
            "distill": int(self.distill),
            "seed": self.seed,
            "eval_token": self.eval_token,
        }

# ==== 5. 记录字符串工具 ====
def _fmt_vec(vals) -> str:
    """
    把 mean/std 元组格式化成紧凑字符串

    参数:
        vals: 序列 (如 (0.5,0.5,0.5) 或 (0.485,0.456,0.406))
    返回:
        str — 三值相同则只写一个 ("0.5"); 否则逗号连接 ("0.485,0.456,0.406")
    """
    vals = tuple(vals)
    # set(vals) 去重后只剩 1 个 -> 三通道同值 -> 压成一个数, 与清单示例格式一致
    if len(set(vals)) == 1:
        return f"{vals[0]:g}"
    return ",".join(f"{v:g}" for v in vals)

# ==== 怎么验证跑对了 ====
# 1) `python -c "from config import ARCH_REGISTRY as R; print(len(R), list(R))"`
#    -> 4 ['r50', 'vits', 'vits1k', 'deitt']
# 2) `python -c "from config import ProtocolCfg as C; print(C(arch='vits',mode='Z2',data=25,seed=42).eid)"`
#    -> vits-z2-25-s42
# 3) `python -c "from config import ProtocolCfg as C; print(C(arch='vits').preproc_str((0.5,)*3,(0.5,)*3))"`
#    -> vit_small/augreg_in21k_ft_in1k/0.5-0.5
# 4) `python -c "from config import ProtocolCfg as C; print(C(mode='Z0').train_aug, C(mode='Z1').train_aug)"`
#    -> False True
# 5) 脏值拦截: `ProtocolCfg(mode='z2')` / `ProtocolCfg(data=33)` 都应抛 ValueError
