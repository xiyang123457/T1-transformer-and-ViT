"""
数据模块 (W3-D1)
- Flowers102 加载 (已手动下载, download=False)
- 图像变换 (统一 ImageNet-1k 归一化)
- 嵌套分层子集抽样
"""
import numpy as np                                    # 数组/随机数 (分层抽样)
import torch                                          # 张量 (预留, 本文件未直接用)
from torch.utils.data import DataLoader, Subset        # 数据加载 / 子集包装 (预留)
import torchvision.datasets as datasets                # 含 Flowers102
import torchvision.transforms as T                     # 图像变换


# 数据根目录: 绝对路径, 不受当前工作目录影响
#   torchvision 会在其下自己拼 "flowers-102/", 所以这里指到 data 层即可
DATA_ROOT = r"d:/learning project/T1 transformer and ViT/data"

# ImageNet-1k 归一化常数
#   三个 backbone (ResNet / ViT / DeiT) 都在 ImageNet 上预训练,
#   输入分布须与预训练一致, 且三者用同一套才可比
IMAGENET_MEAN = (0.485, 0.456, 0.406)                  # 3 通道均值 (R,G,B)
IMAGENET_STD = (0.229, 0.224, 0.225)                   # 3 通道标准差


def build_transform(train: bool):
    """
    构建图像变换

    参数:
        train (bool): True = 训练态 (随机增强); False = 验证/测试态 (确定性)
    返回:
        torchvision.transforms.Compose: 变换流水线
    """
    if train:
        return T.Compose([
            # T.RandomResizedCrop(size): 随机缩放 + 随机位置裁剪到 size x size
            #   作用: 数据增强, 让模型见到不同尺度/位置; 224 对齐预训练输入
            #   注意: 列表元素之间必须有逗号
            T.RandomResizedCrop(224),
            # T.RandomHorizontalFlip(p=0.5): 以 p 概率水平翻转 (默认 p=0.5)
            T.RandomHorizontalFlip(),
            # T.ToTensor(): PIL.Image -> Tensor, 形状 (C,H,W), 值域缩到 [0,1]
            T.ToTensor(),
            # T.Normalize(mean, std): 逐通道执行 (x - mean) / std
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return T.Compose([
        # T.Resize(256): 短边缩到 256, 长边等比缩放
        T.Resize(256),
        # T.CenterCrop(224): 中心裁出 224x224
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataset(split, train_aug=False):
    """
    加载 Flowers102 的某个 split

    参数:
        split (str): 'train' / 'val' / 'test'
        train_aug (bool): 是否使用训练增强 (默认 False)
    返回:
        torchvision Dataset
    """
    # datasets.Flowers102(root, split, transform, download)
    #   root: 数据根目录 (内部拼 flowers-102/)
    #   download=False: 用已手动下载的文件, 不联网
    return datasets.Flowers102(
        root=DATA_ROOT,
        split=split,
        transform=build_transform(train_aug),
        download=False,
    )


def get_targets(dataset):
    """
    取数据集标签数组 (供分层抽样)

    参数:
        dataset: torchvision Dataset
    返回:
        np.ndarray: 标签, 长度 = len(dataset)
    """
    # hasattr(obj, name): 判断对象是否有该属性
    if hasattr(dataset, '_labels'):                    # torchvision 0.24 把标签存 _labels
        # np.asarray: list -> ndarray
        return np.asarray(dataset._labels)
    # 回退 (慢): dataset[i] 返回 (image, label), 逐个取 label
    return np.asarray([y for _, y in dataset])


def stratified_subset_indices(targets, ratio, seed):
    """
    分层抽样出 train 子集索引, 保证嵌套 (同 seed 下小比例是大比例的前缀)

    参数:
        targets (np.ndarray): 每个样本的类别
        ratio (float): 抽样比例, 0 < ratio <= 1
        seed (int): 随机种子
    返回:
        np.ndarray: 抽中的样本索引 (升序)
    """
    # np.random.default_rng(seed): 新建随机数生成器; 同 seed -> 同序列 -> 可复现
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    picked = []                                        # 收集每类抽中的索引
    # np.unique(targets): 返回排序后的唯一类别; 遍历顺序固定 -> 各次调用一致 -> 保证嵌套
    for c in np.unique(targets):
        # np.where(cond)[0]: 返回满足条件的下标 (where 返回元组, 取 [0])
        cls_idx = np.where(targets == c)[0]
        # rng.permutation(x): 返回 x 的随机排列 (不修改原数组)
        perm = rng.permutation(cls_idx)
        # int(round(.)): 四舍五入取整; max(1, .) 至少 1, 防小类被抽空
        k = max(1, int(round(ratio * len(cls_idx))))
        picked.append(perm[:k])                        # 取前 k 个 (前缀)

    # np.concatenate: 各类索引拼成一维; np.sort: 升序, 便于 hash/比较稳定
    return np.sort(np.concatenate(picked))
