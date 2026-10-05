"""
W3-D2 — 训练 CNN teacher (ResNet-50) 于 Flowers102 100% train

用途:
    (1) 维度A 的 CNN 基线;
    (2) W5 DeiT 蒸馏的 teacher。
输入: data/flowers-102 (官方划分: train 1020 / val 1020 / test 6149)
输出:
    - checkpoints/teacher_resnet50.pt (state_dict + config + val_top1)
    - notes/teacher_curves.png (train loss / val top-1 曲线)

跑完应看到的自检数字:
    - 打印 `head: Linear(in_features=2048, out_features=102, bias=True)`  (类别数必须=102)
    - `val_top1` 应落在 0.93 ~ 0.98 (ImageNet 预训练的 ResNet-50 微调 Flowers 的合理区间)
    - 存档文件存在, 约 98MB (ResNet-50 约 25.6M 参数, fp32 约 98MB)
    - notes/teacher_curves.png 存在, 右曲线终点 == 最后一行 val_top1
"""
# ==== 1. 依赖导入 ====
import os                                             # 建 checkpoints 目录 (os.makedirs)
import time                                           # 计时, 打印单 epoch 秒数
import copy                                           # 深拷贝最优权重 (避免被后续训练覆盖)
import numpy as np                                    # 设 numpy 随机种子 (与 torch 双管齐下)
import torch                                          # 张量 / 训练 / 存档
import torch.nn as nn                                 # CrossEntropyLoss / Linear
from torch.utils.data import DataLoader, Subset        # DataLoader 批加载 / Subset 取子集
import torchvision.models as models                    # 取 ResNet-50 及预训练权重枚举
from data import get_dataset, get_targets, stratified_subset_indices  # 复用 W3-D1 数据模块
from plot_utils import plot_history                    # 训练曲线绘制 (边角料工具)

# ==== 2. 全局配置与设备 (受控实验的全部常数, 全部写死) ====
SEED = 42
# 变量 SEED: int, 全局随机种子
#   示例值: 42
#   为什么是 42: 只求"固定且好记"; 全项目所有实验必须用同一个种子才可比
NUM_CLASSES = 102
# 变量 NUM_CLASSES: int, 分类头类别数
#   示例值: 102 (Flower102 的类别数)
#   为什么: 预训练头默认 1000 类, 不改成 102 则输出维度错、loss 无意义
EPOCHS = 30
# 变量 EPOCHS: int, 最大训练轮数 (上界, 实际由早停决定何时停)
#   示例值: 30
#   为什么: train 只有 1020 张, 30 轮足够收敛; 设大以免早停前就撞上限
BATCH = 32
# 变量 BATCH: int, 单批样本数
#   示例值: 32 (ResNet-50 在 8GB 显存下 224x224 + amp 的经验值)
#   为什么: 显存不够就降到 16; 太小会拖慢且 BN 统计不稳
LR_BACKBONE = 1e-4
# 变量 LR_BACKBONE: float, 预训练骨干的学习率 (小)
#   示例值: 1e-4
#   为什么: 骨干已预训练, 大 lr 会破坏预训练特征 (灾难性遗忘)
LR_HEAD = 1e-3
# 变量 LR_HEAD: float, 新分类头的学习率 (大)
#   示例值: 1e-3 (= 骨干的 10 倍)
#   为什么: 头是随机初始化的, 需要更快学到映射; 两头分开是"分层 lr"的核心
WEIGHT_DECAY = 1e-4
# 变量 WEIGHT_DECAY: float, AdamW 的权重衰减
#   示例值: 1e-4
#   为什么: 微调常用值, 抑制过拟合
PATIENCE = 5
# 变量 PATIENCE: int, 早停耐心值
#   示例值: 5
#   为什么: val_top1 连续 5 轮不升则停; 太小会误停, 太大浪费算力
NUM_WORKERS = 0
# 变量 NUM_WORKERS: int, DataLoader 子进程数
#   示例值: 0
#   为什么: Windows 下多进程易出 spawn/句柄问题, 先 0 跑通, 稳定后再试 2/4
PROJECT_ROOT = r"d:/learning project/T1 transformer and ViT"
# 变量 PROJECT_ROOT: str, 项目根目录 (绝对路径, 原始字符串)
#   示例值: "d:/learning project/T1 transformer and ViT"
#   为什么: IO 路径一律基于它拼绝对路径 -> 从任何目录运行都存到同一处 (避免上次存进 data/)
OUT_PATH = PROJECT_ROOT + "/checkpoints/teacher_resnet50.pt"
CURVE_PATH = PROJECT_ROOT + "/notes/teacher_curves.png"
# 变量 CURVE_PATH: str, 曲线图保存路径; notes/ 是入库目录, 图作为报告素材留存
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 变量 DEVICE: str, 计算设备
#   示例值: "cuda"
#   为什么: 有条件就用 GPU; 后续 .to(DEVICE) 统一搬运张量/模型
AMP = (DEVICE == "cuda")
# 变量 AMP: bool, 是否启用混合精度
#   示例值: True (仅 CUDA)
#   为什么: CPU 不支持 CUDA autocast; 出 NaN 时手动改 False 排查

# ==== 3. 模型构建 ====
def build_teacher(num_classes=NUM_CLASSES):
    """
    构建 ResNet-50 teacher, 分类头换成 num_classes
    参数:
        num_classes (int): 分类数, Flowers102 = 102
    返回:
        nn.Module: 可直接 .to(DEVICE) 的模型
    """
    # models.resnet50(weights=...) -> nn.Module
    #   作用: 取 ResNet-50 结构, 并加载 ImageNet-1k V2 预训练权重
    #   关键参数: weights=ResNet50_Weights.IMAGENET1K_V2 (torchvision 官方最优默认权重)
    #   坑: 返回的是 1000 类头, 必须换掉, 否则结果全错
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    # 变量 in_features: int, 最后一层全连接的输入维
    #   示例值: 2048 (ResNet-50 的瓶颈维)
    #   为什么取它: 换头时输出维要 = 102, 输入维必须沿用原 2048
    # nn.Linear(2048, 102) -> 替换原头
    #   作用: 把 1000 类头换成 102 类头
    #   关键参数: in=2048 (沿用), out=102 (Flowers)
    #   坑: 新头是随机初始化的, 所以要用大 lr (见 LR_HEAD)
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ==== 4. 数据管道 (train 100% + val 固定) ====
def build_loaders():
    """
    构建 train(100%子集) / val 两个 DataLoader
    返回:
        (train_loader, val_loader, train_full)
    """
    train_full = get_dataset('train', train_aug=True)
    # 变量 train_full: torchvision Dataset, 训练集(带随机增强)
    #   示例值: len=1020
    #   为什么 train_aug=True: 训练要增强; 验证必须 False, 否则评估不可比
    val_set = get_dataset('val', train_aug=False)
    # 变量 val_set: Dataset, 验证集(确定性变换), len=1020; 全程固定, 是"受控"的基准
    idx100 = stratified_subset_indices(get_targets(train_full), 1.0, SEED)
    # 变量 idx100: np.ndarray, 100% 子集索引
    #   示例值: 长度 1020 的升序索引
    #   为什么用 1.0: 与协议口径统一, 之后换 0.1/0.25/0.5 只改这一个比例数字
    train_set = Subset(train_full, idx100)
    # 变量 train_set: Subset, 用索引包装后的训练集; 为什么不直接用它 —— 为了口径统一
    # DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)
    #   作用: 分批迭代数据
    #   关键参数: 训练 shuffle=True (打散); 验证 shuffle=False (顺序稳定); pin_memory 加速 H2D
    #   坑: num_workers=0 时只能在主进程取数据, 慢但不报错; Windows 先保持 0
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, train_full

# ==== 5. 优化器与损失 (分层 lr) ====
def build_optimizer(model):
    """
    分层 lr 优化器: 骨干小 lr, 分类头大 lr
    参数:
        model (nn.Module)
    返回:
        torch.optim.Optimizer
    """
    head_params = list(model.fc.parameters())
    # 变量 head_params: list[Tensor], 分类头参数
    #   示例值: [weight(102,2048), bias(102)]
    #   为什么单独拎出: 要给它们单独设大 lr
    head_ids = {id(p) for p in head_params}
    # 变量 head_ids: set[int], 头参数的 id 集合
    #   示例值: {140..., 141...}
    #   为什么用 id: Tensor 做集合元素不可靠, 用对象 id 判重最稳
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    # 变量 backbone_params: list[Tensor], 除头以外的全部参数; 与之互补, 不重不漏
    # torch.optim.AdamW(param_groups, weight_decay) -> Optimizer
    #   作用: AdamW 优化器; 传参数组列表可为不同组设不同 lr
    #   关键参数: 每组一个 dict {"params":..., "lr":...}; weight_decay 统一
    #   坑: 忘加 weight_decay 会退化成 Adam; lr 写反(骨干大、头小)会毁掉预训练
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    return optimizer

# ==== 6. 评估 (val top-1) ====
@torch.no_grad()
def evaluate(model, loader):
    """
    在给定 loader 上算 top-1 准确率
    参数:
        model (nn.Module)
        loader (DataLoader)
    返回:
        acc (float): top-1 准确率, 范围 [0,1]
    """
    # @torch.no_grad(): 装饰器, 整函数关闭梯度记录
    #   为什么: 评估不反传, 关掉可省显存、加速
    # model.eval() -> None
    #   作用: 切评估态 (关 dropout, BN 用滑动统计)
    #   坑: 忘调会用训练态统计, 结果偏且不稳定
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        logits = model(x)                     # (B, 102)
        # logits.argmax(dim=1) -> (B,)
        #   作用: 取每个样本最大 logit 的下标作预测类别
        #   关键参数: dim=1 在类别维上取最大 (dim=0 是 batch 维, 错)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

# ==== 7. 训练一个 epoch ====
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    """
    训练一个 epoch
    参数:
        model, loader, optimizer, scaler (GradScaler), criterion (损失)
    返回:
        avg_loss (float): 样本加权平均损失
    """
    model.train()                              # 训练态: 开 dropout / BN 更新统计
    running = 0.0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # 以上: set_to_none=True 比置 0 更省显存
        # torch.amp.autocast('cuda', enabled=AMP) -> 上下文
        #   作用: 在其内前向自动用半精度 (fp16) 加速
        #   关键参数: enabled=AMP 控制开关; 数值敏感算子仍自动回退 fp32
        #   坑: 出 NaN 先把 AMP 关掉定位
        with torch.amp.autocast('cuda', enabled=AMP):
            logits = model(x)                  # (B, 102)
            # nn.CrossEntropyLoss()(logits, y) -> scalar
            #   作用: 多分类交叉熵 (内部含 log_softmax + NLL)
            #   关键参数: logits 是未归一化分数 (不要自己先 softmax), y 是类别下标
            #   坑: 传已 softmax 的概率会算错
            loss = criterion(logits, y)
        # scaler.scale(loss).backward() -> None
        #   作用: 混合精度下把 loss 放大再反传, 防小梯度下溢成 0
        #   坑: 用 AMP 时必须经 scaler, 不能直接 loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)                 # scaler 控制下更新参数 (含跳过 inf/nan step)
        scaler.update()                        # 调整下一轮的缩放因子
        running += loss.item() * y.numel()     # 按样本数加权, 避免最后一批偏小影响
    return running / len(loader.dataset)

# ==== 8. 主流程: 训练循环 + 早停 + 存档 + 画曲线 ====
def main():
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    # 以上: 同时固定 numpy 与 torch 的种子, 保证两次运行结果一致
    # os.makedirs(path, exist_ok=True): 递归建目录, 已存在不报错
    #   为什么两个都建: OUT_PATH / CURVE_PATH 是绝对路径, 对应目录可能还不存在
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CURVE_PATH), exist_ok=True)
    print(f"device={DEVICE} | amp={AMP}")

    model = build_teacher().to(DEVICE)
    print("head:", model.fc)                    # 必须看到 2048->102

    train_loader, val_loader, train_full = build_loaders()
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    # 变量 scaler: GradScaler, 混合精度的梯度缩放器; 只有 AMP 时才真正生效

    history = {"epoch": [], "train_loss": [], "val_top1": []}
    # 变量 history: dict[str, list], 逐轮记录三个等长序列
    #   示例值: {"epoch":[0,1,...], "train_loss":[4.6,...], "val_top1":[0.12,...]}
    #   为什么用 dict: 三条序列共享同一 x 轴(epoch), 画图时直接取用

    best_acc, best_epoch, best_state, bad = 0.0, -1, None, 0
    # 变量 best_acc: float, 历史最优 val top-1
    #   示例值: 0.0 -> 逐渐升到 ~0.96
    #   为什么用 copy.deepcopy 存 state: 训练会持续改权重, 必须快照最优那一刻
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        acc = evaluate(model, val_loader)       # 固定 val 上评估
        print(f"[{epoch:02d}] loss={loss:.4f} val_top1={acc:.4f} ({time.time()-t0:.1f}s)")

        # 逐轮记录, 供最后画曲线
        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["val_top1"].append(acc)

        if acc > best_acc:                      # 刷新最优
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:                 # 早停
                print(f"early stop @ epoch {epoch} (best {best_acc:.4f} @ {best_epoch})")
                break

    # torch.save(obj, path) -> None
    #   作用: 序列化到磁盘; 这里存 dict (权重 + config + 数字)
    #   坑: 只存权重不存 config, W5 蒸馏时分不清归一化常数/类别顺序 -> 对不齐 (坑 9)
    ckpt = {
        "model": best_state,
        "config": {
            "arch": "resnet50",
            "num_classes": NUM_CLASSES,
            "imagenet_mean": (0.485, 0.456, 0.406),
            "imagenet_std":  (0.229, 0.224, 0.225),
            "classes": train_full.classes,      # index -> 类名, W5 蒸馏必须对齐
            "input_size": 224,
        },
        "val_top1": best_acc,
        "epoch": best_epoch,
    }
    torch.save(ckpt, OUT_PATH)
    print(f"saved: {OUT_PATH} | best val_top1={best_acc:.4f} @ epoch {best_epoch}")

    # 画训练曲线 (loss / val top-1)
    # plot_history(history, out_path) -> None
    #   作用: 把 history 画成两子图并存 PNG
    #   关键参数: out_path 所在目录 (notes/) 必须已存在
    #   坑: history 为空 (0 轮) 时画图会得到空图, 但不报错
    plot_history(history, CURVE_PATH)


if __name__ == "__main__":
    main()

# ==== 怎么验证跑对了 ====
# 1) 开跑即应打印 `head: Linear(in_features=2048, out_features=102, bias=True)` -> 类别数对
#    (若显示 out_features=1000, 说明没换头, 结果全废)
# 2) 每轮 val_top1 应单调爬升后趋稳, 最终落在 0.93 ~ 0.98
#    (若一直 ~0.01 = 1/102, 说明头没换或标签错位)
# 3) 结束打印 `saved: ... | best val_top1=0.9xxx @ epoch E`, 且 checkpoints/teacher_resnet50.pt 存在 (约 98MB)
# 4) notes/teacher_curves.png 存在, 右曲线终点数值 == 日志最后一行的 val_top1
# 5) 复跑一次 (同种子), best val_top1 应与上次完全一致 -> 可复现
# 6) loss 出现 nan -> 先把 AMP 改成 False 重跑, 若消失即为混合精度数值问题
