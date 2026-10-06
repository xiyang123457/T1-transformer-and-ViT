"""
W4-D1 — 数据模块: 按架构官方预处理 + test 拦截 + 四档子集索引读写

用途:
    给 protocol / run_matrix 提供「与架构匹配的输入分布」(口径 3) 和「可复现的 train 子集」(口径 16⑤)。
输入: ArchSpec(来自 config) / 已建好的 model / split 名
输出: Dataset、四档子集索引、该架构的预处理常数 (供 preproc 列落盘)

跑完应看到的自检数字 (见文件末尾「怎么验证跑对了」):
    - ViT-S  : mean=std=(0.5,0.5,0.5), crop_pct=0.9, interpolation=bicubic
    - DeiT   : mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
    - R50    : mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
    - get_dataset('test', ...) 抛 RuntimeError
    - 四档子集大小 == 102 / 204 / 510 / 1020, 且嵌套成立
"""
# ==== 1. 依赖导入 ====
import hashlib                                          # 算子集索引指纹 (可复现校验)
import json                                             # 读写 subsets/tier_*.json
import os                                               # 目录操作
import numpy as np                                      # 数组/随机数 (分层抽样)
import torchvision.datasets as datasets                  # 含 Flowers102
import torchvision.transforms as T                       # torchvision 变换 (ResNet 用)

# ==== 2. 路径与常数 ====
PROJECT_ROOT = r"d:/learning project/T1 transformer and ViT"
# 变量 PROJECT_ROOT: str, 项目根 (绝对路径, 原始字符串)
#   为什么: IO 路径一律基于它 -> 从任何目录运行都落到同一处

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
# 变量 DATA_ROOT: str, 数据集根; torchvision 会在其下自己拼 "flowers-102/"
#   示例值: "d:/learning project/T1 transformer and ViT/data"

SUBSETS_DIR = os.path.join(PROJECT_ROOT, "subsets")
# 变量 SUBSETS_DIR: str, 四档子集索引目录 (口径 16⑤ 不可逆保险)
#   示例值: ".../subsets"; 内含 tier_10/25/50/100.json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
# 变量 IMAGENET_MEAN: tuple[float,float,float], ImageNet-1k 三通道均值
#   为什么: ResNet-50(V2) 与 DeiT-Ti 官方都用它; ViT-S **不用**它 (是 0.5) —— 口径 3 的核心
IMAGENET_STD = (0.229, 0.224, 0.225)
# 变量 IMAGENET_STD: tuple[float,float,float], ImageNet-1k 三通道标准差

# ==== 3. 图像变换 (按架构; 口径 3: 每架构用官方预处理) ====
def timm_transform(model, train: bool):
    """
    从 timm 模型解析官方预处理并构建变换

    参数:
        model (nn.Module): timm 模型 (需已建好; 只读它的 pretrained_cfg)
        train (bool): True = 训练增强; False = 确定性中心裁剪
    返回:
        (transform, data_cfg: dict) — data_cfg 含 mean/std/crop_pct/interpolation/input_size
    """
    import timm
    # timm.data.resolve_data_config({}, model=model) -> dict
    #   作用: 把模型自带的 pretrained_cfg 解析成实际数据配置
    #   关键参数: 第一个 {} 是用户覆盖项 (这里不覆盖, 完全听官方)
    #   坑: 记录必须用"这里解析出来的"值, 而不是手抄的常数 (timm 升级会变)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    # timm.data.create_transform(...) -> torchvision.transforms.Compose
    #   作用: 按解析出的 mean/std/crop_pct/interpolation 构造标准变换
    #   关键参数: is_training=True 时含 RandomResizedCrop + 水平翻转 (即 "basic" 增强)
    #   坑: 不传 auto_augment/re_prob 时才是 basic; 传了就成 strong (那是 W5 维度 D)
    transform = timm.data.create_transform(
        input_size=data_cfg["input_size"], is_training=train,
        mean=data_cfg["mean"], std=data_cfg["std"],
        crop_pct=data_cfg.get("crop_pct", 0.875),
        interpolation=data_cfg.get("interpolation", "bilinear"),
    )
    return transform, data_cfg

def torchvision_transform(train: bool):
    """
    torchvision 架构 (ResNet-50) 的标准 ImageNet 变换

    参数:
        train (bool): True = 随机裁剪 + 翻转; False = Resize256 + CenterCrop224
    返回:
        (transform, data_cfg: dict)
    """
    if train:
        transform = T.Compose([
            # T.RandomResizedCrop(224): 随机缩放+位置裁剪到 224 (增强)
            T.RandomResizedCrop(224),
            # T.RandomHorizontalFlip(): p=0.5 水平翻转
            T.RandomHorizontalFlip(),
            T.ToTensor(),                                    # PIL -> Tensor(C,H,W), [0,1]
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),        # 逐通道 (x-mean)/std
        ])
    else:
        transform = T.Compose([
            T.Resize(256),                                   # 短边缩到 256
            T.CenterCrop(224),                               # 中心裁 224 (crop_pct=0.875)
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    data_cfg = {"input_size": (3, 224, 224), "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "crop_pct": 0.875, "interpolation": "bilinear"}
    return transform, data_cfg

def build_transform(spec, train: bool, model=None):
    """
    按架构分派构建变换

    参数:
        spec (ArchSpec): 架构注册项 (决定走 timm 还是 torchvision)
        train (bool): 是否训练增强
        model (nn.Module|None): timm 架构必需 (用它解析官方 mean/std)
    返回:
        (transform, data_cfg: dict)
    """
    if spec.library == "timm":
        if model is None:
            raise ValueError("timm 架构必须传入 model (用它解析官方预处理常数)")
        return timm_transform(model, train)
    return torchvision_transform(train)

# ==== 4. 数据集加载 (口径 7: test 在 W8 之前不可访问) ====
def get_dataset(split, transform):
    """
    加载 Flowers102 的某个 split

    参数:
        split (str): 'train' / 'val' / 'test'
        transform: 图像变换 (来自 build_transform)
    返回:
        torchvision Dataset
    """
    # 设计决策: 把 "禁止 test" 写成硬拦断 (而不是靠纪律自觉)
    #   原因: 口径 7 —— test 一旦反复看就退化成 val, 整个实验的独立性就没了
    if split == "test":
        raise RuntimeError("test 集在 W8 定稿前不可访问 (口径 7)")
    if split not in ("train", "val"):
        raise ValueError(f"未知 split={split}; 应为 'train' 或 'val'")
    # datasets.Flowers102(root, split, transform, download=False) -> Dataset
    #   作用: 读本地已下载的 flowers-102 (torchvision 内部会拼 root/flowers-102/)
    #   坑: download=True 会联网; 这里必须 False (数据已手动下载)
    return datasets.Flowers102(root=DATA_ROOT, split=split, transform=transform, download=False)

def get_raw_dataset(split):
    """加载不带动画的原始 split (只取标签用, 供子集抽样)"""
    if split == "test":
        raise RuntimeError("test 集在 W8 定稿前不可访问 (口径 7)")
    return datasets.Flowers102(root=DATA_ROOT, split=split, transform=None, download=False)

def get_targets(dataset):
    """
    取数据集标签数组 (供分层抽样)

    参数:
        dataset: torchvision Dataset
    返回:
        np.ndarray[int64]: 标签, 长度 = len(dataset)
    """
    if hasattr(dataset, "_labels"):            # torchvision 0.24 把标签存在 _labels
        return np.asarray(dataset._labels)
    return np.asarray([y for _, y in dataset])   # 回退 (慢)

# ==== 5. 嵌套分层子集抽样 (沿用 W3; 保证 10%⊂25%⊂50%⊂100%) ====
def stratified_pick_per_class(targets, ratio, seed):
    """
    按类分层抽取, 返回"每类抽中的索引"

    参数:
        targets (np.ndarray): 每个样本的类别
        ratio (float): 抽样比例, 0 < ratio <= 1
        seed (int): 随机种子
    返回:
        dict[int, list[int]]: {类别: [该类抽中的全局索引, 升序]}
    """
    # np.random.default_rng(seed): 同 seed -> 同一随机序列 -> 可复现
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    picked: dict[int, list[int]] = {}
    # np.unique(targets): 升序唯一类别; 遍历顺序固定 -> 嵌套性质成立
    for c in np.unique(targets):
        # np.where(cond)[0]: 取满足条件的下标 (where 返回元组, 取 [0])
        cls_idx = np.where(targets == c)[0]
        # rng.permutation(x): x 的随机排列 (不改原数组)
        perm = rng.permutation(cls_idx)
        # max(1, round(...)): 至少 1 个, 防小类被抽空
        k = max(1, int(round(ratio * len(cls_idx))))
        picked[int(c)] = sorted(int(i) for i in perm[:k])   # 取前缀 -> 小比例是大比例子集
    return picked

def stratified_subset_indices(targets, ratio, seed):
    """
    分层抽样出 train 子集索引 (扁平, 升序)

    参数:
        targets (np.ndarray): 类别; ratio (float): 比例; seed (int): 种子
    返回:
        np.ndarray[int64]: 抽中的索引 (升序)
    """
    per_class = stratified_pick_per_class(targets, ratio, seed)
    flat = np.concatenate([np.asarray(v, dtype=np.int64) for v in per_class.values()])
    return np.sort(flat)

def index_hash(idx) -> str:
    """
    子集索引指纹

    参数:
        idx: 索引序列
    返回:
        str — md5 十六进制; 固定转 int64 保证同一集合永远同一 hash
    """
    arr = np.asarray(idx, dtype=np.int64)      # 固定 dtype: 否则 tobytes 结果会不一致
    return hashlib.md5(arr.tobytes()).hexdigest()

# ==== 6. 四档子集索引落盘 / 读取 (口径 16⑤: 开跑前必须补上) ====
def tier_path(data: int) -> str:
    """subsets/tier_{p}.json 的绝对路径"""
    return os.path.join(SUBSETS_DIR, f"tier_{data}.json")

def save_tier(data: int, targets, seed: int):
    """
    生成并落盘某一档子集索引 (由 make_subsets.py 调用)

    参数:
        data (int): 档位百分比 (10/25/50/100)
        targets (np.ndarray): train 标签
        seed (int): 抽样种子
    返回:
        dict — 落盘的完整记录
    """
    ratio = data / 100.0
    per_class = stratified_pick_per_class(targets, ratio, seed)
    flat = np.sort(np.concatenate([np.asarray(v, dtype=np.int64) for v in per_class.values()]))
    record = {
        "tier": data,
        "ratio": ratio,
        "seed": seed,
        "n_classes": len(per_class),
        "n_total": int(flat.size),
        "hash": index_hash(flat),                         # 索引指纹 (校验用)
        "per_class": {str(k): v for k, v in per_class.items()},   # 每类抽中的 index 列表
        "indices": [int(i) for i in flat],                # 扁平索引 (训练时直接用)
    }
    os.makedirs(SUBSETS_DIR, exist_ok=True)
    with open(tier_path(data), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return record

def load_tier_indices(data: int, seed: int, targets=None, strict: bool = True):
    """
    读取某一档子集索引, 并做两级校验

    参数:
        data (int): 档位百分比
        seed (int): 期望的抽样种子 (与落盘时不一致 -> 说明口径变了, 直接报错)
        targets (np.ndarray|None): train 标签; 给了就重算一遍做"漂移校验"
        strict (bool): 漂移时是否直接抛错 (默认 True)
    返回:
        (np.ndarray[int64] 索引, dict 记录)
    """
    path = tier_path(data)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少子集索引 {path}; 请先跑 make_subsets.py (口径 16⑤)")
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    idx = np.asarray(rec["indices"], dtype=np.int64)
    # 校验 1: 文件自身完整性 —— 记录的 hash 必须等于对 indices 重算的 hash
    if index_hash(idx) != rec["hash"]:
        raise RuntimeError(f"{path} 内容与 hash 不一致 (文件被改动过?)")
    # 校验 2: 种子一致
    if int(rec["seed"]) != int(seed):
        raise RuntimeError(f"{path} 的 seed={rec['seed']} 与本次 seed={seed} 不一致")
    # 校验 3: 漂移检测 —— 用当前代码重算, 若与落盘不一致说明抽样逻辑变了
    if targets is not None:
        recomputed = stratified_subset_indices(targets, rec["ratio"], rec["seed"])
        if index_hash(recomputed) != rec["hash"]:
            msg = f"tier_{data} 子集与当前抽样代码不一致 (漂移!); 重跑将换子集 -> 跨组不可比"
            if strict:
                raise RuntimeError(msg)
            print("[WARN]", msg)
    return idx, rec

# ==== 怎么验证跑对了 ====
# 1) `python -c "from data import get_raw_dataset, get_targets, stratified_subset_indices as S, index_hash; t=get_targets(get_raw_dataset('train')); print([len(S(t,p/100,42)) for p in (10,25,50,100)])"`
#    -> [102, 204, 510, 1020]
# 2) 嵌套: S(t,.10,42) ⊆ S(t,.25,42) ⊆ S(t,.50,42) ⊆ S(t,1.0,42) (set 包含关系全 True)
# 3) 复现: 同 seed 调两次 index_hash 相同
# 4) `data.get_dataset('test', None)` -> RuntimeError: test 集在 W8 定稿前不可访问
# 5) 跑完 make_subsets.py 后: subsets/ 下有 tier_10/25/50/100.json, 且 load_tier_indices 三校验全过
