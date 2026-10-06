"""
W3-D3 — 迁移协议 run_protocol(cfg): Z0/Z1/Z2 统一入口

三档本质 (文档知识点 3):
    Z0 冻结+NCM : 无梯度提特征, 最近类均值分类 -> 检验"预训练特征本身好不好"
    Z1 只训头   : 冻结骨干, 只训线性头         -> 检验"学一个线性映射够不够"
    Z2 全参微调 : 解冻全量                     -> 检验"要不要改造特征本身"
输入: ProtocolCfg
输出: val_top1 (float), 供 W4 矩阵填表

跑完应看到的自检数字:
    - Z0: 权重与跑之前逐元素相等 (torch.equal 全 True)
    - Z1: 可训练参数个数 == 头的参数个数
    - Z2: 至少 conv1 (或 patch_embed) 权重发生变化
"""
# ==== 1. 依赖导入 ====
import time                                            # 计时 (D5 要测单 epoch 秒数)
import numpy as np                                     # 设种子 (与 torch 双管齐下)
import torch                                           # 张量/训练
import torch.nn as nn                                  # 损失层
from torch.utils.data import DataLoader, Subset         # 批加载 / 子集
from data import get_dataset, get_targets, stratified_subset_indices   # W3-D1 数据模块
from config import ProtocolCfg                          # W3-D3 配置

# ==== 2. 模型构建 (三种架构统一入口) ====
def build_model(cfg):
    """
    按 cfg.arch 建模型, 分类头换成 cfg.num_classes
    参数:
        cfg (ProtocolCfg): 含 arch / num_classes
    返回:
        nn.Module
    """
    if cfg.arch == "resnet50":
        import torchvision.models as models
        # models.resnet50(weights=IMAGENET1K_V2) -> nn.Module
        #   作用: ResNet-50 结构 + ImageNet-1k V2 预训练权重
        #   关键参数: weights 取官方枚举, 保证权重来源确定
        #   坑: 原头 1000 类, 必须换
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)   # 2048 -> 102
        return m
    import timm
    # timm.create_model(name, pretrained=True, num_classes=102) -> nn.Module
    #   作用: 按名字建 ViT / DeiT 并加载预训练
    #   关键参数: num_classes 会同时改 DeiT 的 head 与 head_dist (坑 4)
    #   坑: 不传 num_classes 会得到 1000 类头, 结果全错
    return timm.create_model(cfg.arch, pretrained=True, num_classes=cfg.num_classes)

def head_params(model):
    """
    取分类头参数 (不同架构名字不同)
    参数:
        model (nn.Module)
    返回:
        list[Tensor]: 头参数列表
    """
    if hasattr(model, "fc"):        # torchvision ResNet 用 fc
        return list(model.fc.parameters())
    return list(model.head.parameters())   # timm ViT/DeiT 用 head

# ==== 3. 特征提取 (Z0 专用) ====
def extract_features(model, loader, device):
    """
    冻结前向, 截获"分类头之前"的特征
    参数:
        model (nn.Module)
        loader (DataLoader)
        device (str)
    返回:
        (feats (N,D) tensor, labels (N,) tensor)   # 都在 CPU 上
    """
    # register_forward_hook -> RemovableHandle
    #   作用: 在 head 前向时触发, 用 inp[0] 拿到 head 的输入 = backbone 输出特征
    #   关键参数: hook(module, input, output); input 是元组, 取 [0]
    #   坑: 用完必须 handle.remove(), 否则 hook 残留会污染后续前向
    head = model.fc if hasattr(model, "fc") else model.head
    box = {}
    handle = head.register_forward_hook(
        lambda m, inp, out: box.__setitem__("f", inp[0].detach()))
    feats, labels = [], []
    model.eval()                                  # 评估态: BN 用滑动统计、关 dropout
    with torch.no_grad():                         # 提特征不需要梯度
        for x, y in loader:
            model(x.to(device))                   # 触发 hook, 结果写进 box["f"]
            feats.append(box["f"].cpu())
            labels.append(y)
    handle.remove()                               # 移除 hook
    return torch.cat(feats), torch.cat(labels)

# ==== 4. Z0: 冻结 + NCM (最近类均值, 无训练) ====
def run_Z0(model, train_loader, val_loader, device):
    """
    零训练协议: 提特征 -> 每类均值 -> 最近类均值分类
    参数:
        model, train_loader, val_loader, device
    返回:
        val_top1 (float)
    """
    f_tr, y_tr = extract_features(model, train_loader, device)   # (N_tr, D)
    f_va, y_va = extract_features(model, val_loader, device)     # (N_va, D)
    # 逐行 L2 归一化
    #   作用: 让距离只比"方向"不比"模长", NCM 常规
    #   关键参数: dim=1 沿特征维归一化; keepdim=True 保持 (N,1) 便于广播相除
    f_tr = f_tr / f_tr.norm(dim=1, keepdim=True)
    f_va = f_va / f_va.norm(dim=1, keepdim=True)
    classes = torch.unique(y_tr)                                 # (C,) 升序类别
    # torch.stack([...]) -> (C, D)
    #   作用: 每类取均值向量作为该类"中心"
    means = torch.stack([f_tr[y_tr == c].mean(0) for c in classes])
    # torch.cdist(A, B) -> (N_va, C)
    #   作用: 成对欧氏距离; 每个 val 样本到每个类中心的距离
    #   坑: A/B 需同为 float, 且维度一致
    dist = torch.cdist(f_va, means)
    pred = classes[dist.argmin(1)]                               # 取最近中心的下标
    return (pred == y_va).float().mean().item()                  # val top-1

# ==== 5. 训练/评估循环 (Z1 / Z2 共用) ====
def train_and_eval(model, tr_loader, va_loader, optimizer, device, cfg):
    """
    通用训练 + 早停 (Z1/Z2 共用)
    参数:
        model, tr_loader, va_loader, optimizer, device, cfg
    返回:
        best_acc (float)
    """
    criterion = nn.CrossEntropyLoss()          # 多分类交叉熵 (联合 log_softmax+NLL)
    best, bad = 0.0, 0
    # 变量 best: float, 历史最优 val top-1; 示例值 0.0 -> ~0.96
    # 变量 bad: int, 连续未提升轮数; 达到 cfg.patience 即早停
    for epoch in range(cfg.epochs):
        t0 = time.time()
        # 变量 t0: float, 本轮起始时间戳 (time.time() 的返回值)
        #   为什么: D5 要用"单 epoch 秒数"推算整个矩阵的算力预算, 必须逐轮计时
        model.train()                          # 训练态
        running = 0.0
        # 变量 running: float, 本轮累计损失
        #   示例值: 0.0 -> 累加到 4.6 * 102 之类
        #   为什么按样本数加权: 最后一批常不足 batch, 不加权会拉偏均值
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)   # 清梯度 (置 None 比置 0 省显存)
            loss = criterion(model(x), y)
            loss.backward()                    # 反传
            optimizer.step()                   # 更新
            running += loss.item() * y.numel()      # 按样本数加权累计
        tr_loss = running / len(tr_loader.dataset)
        # ---- 评估 ----
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in va_loader:
                pred = model(x.to(device)).argmax(1)     # 取最大 logit 类别
                correct += (pred == y.to(device)).sum().item()
                total += y.numel()
        acc = correct / total
        # 同时打印 train loss 与 val: 才能判断"是否在学" (D4 冒烟要求 loss 下降)
        # 末尾 (Xs): 单 epoch 耗时, D5 预算表直接引用这个数
        print(f"  [{epoch:02d}] train_loss={tr_loss:.4f} val_top1={acc:.4f} ({time.time()-t0:.1f}s)")
        if acc > best:
            best, bad = acc, 0
        else:
            bad += 1
            if bad >= cfg.patience:            # 早停
                break
    return best

# ==== 6. Z1: 只训头 (线性探针) ====
def run_Z1(model, tr_loader, va_loader, device, cfg):
    """冻结骨干, 只训头"""
    for p in model.parameters():
        p.requires_grad = False                # 全冻结 (不建梯度, 省显存)
    hp = head_params(model)
    for p in hp:
        p.requires_grad = True                 # 只解冻头
    # torch.optim.AdamW(params, lr, weight_decay) -> Optimizer
    #   作用: 只优化头参数
    #   关键参数: lr 用大 lr (cfg.lr_head)
    #   坑: 若忘了解冻头, 优化器里全是 requires_grad=False 的参数 -> 不更新, 精度卡死
    opt = torch.optim.AdamW(hp, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    return train_and_eval(model, tr_loader, va_loader, opt, device, cfg)

# ==== 7. Z2: 全参微调 (分层 lr) ====
def run_Z2(model, tr_loader, va_loader, device, cfg):
    """解冻全量, 骨干小 lr + 头大 lr"""
    for p in model.parameters():
        p.requires_grad = True                 # 全解冻
    hp = head_params(model); hids = {id(p) for p in hp}
    bb = [p for p in model.parameters() if id(p) not in hids]    # 骨干参数
    # AdamW(param_groups) -> Optimizer
    #   作用: 两组不同 lr (骨干小、头大)
    #   关键参数: 每组 dict {"params","lr"}
    #   坑: 两组若重叠会重复更新; 用 id 判重保证互补
    opt = torch.optim.AdamW([
        {"params": bb, "lr": cfg.lr_backbone},
        {"params": hp, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    return train_and_eval(model, tr_loader, va_loader, opt, device, cfg)

# ==== 8. 统一入口 ====
def run_protocol(cfg: ProtocolCfg):
    """
    按 cfg.mode 分派 Z0/Z1/Z2
    参数:
        cfg (ProtocolCfg)
    返回:
        val_top1 (float)
    """
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)   # 固定种子 (可复现)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)

    # Z0 只提特征 -> 不需要增强; Z1/Z2 要增强
    train_ds = get_dataset("train", train_aug=(cfg.mode != "Z0"))
    val_ds = get_dataset("val", train_aug=False)
    # stratified_subset_indices(targets, ratio, seed) -> np.ndarray
    #   作用: 取该比例的嵌套子集索引
    #   关键参数: ratio 来自 cfg (数据量维), seed 固定 -> 与 D1 一致
    idx = stratified_subset_indices(get_targets(train_ds), cfg.ratio, cfg.seed)
    tr_loader = DataLoader(Subset(train_ds, idx), batch_size=cfg.batch,
                           shuffle=True, num_workers=cfg.num_workers)
    va_loader = DataLoader(val_ds, batch_size=cfg.batch,
                           shuffle=False, num_workers=cfg.num_workers)

    if cfg.mode == "Z0": return run_Z0(model, tr_loader, va_loader, device)
    if cfg.mode == "Z1": return run_Z1(model, tr_loader, va_loader, device, cfg)
    if cfg.mode == "Z2": return run_Z2(model, tr_loader, va_loader, device, cfg)
    raise ValueError(f"未知 mode: {cfg.mode}")

# ==== 怎么验证跑对了 ====
# 1) Z0 跑完: 与跑前逐元素 torch.equal 全 True (没训练); val_top1 应在 0.8 以上 (特征够好)
# 2) Z1 跑完: sum(p.requires_grad) == 头参数量; val_top1 应明显高于 Z0
# 3) Z2 跑完: model.conv1.weight 与跑前不同; val_top1 应最高
# 4) 三档 val_top1 预期排序: Z0 < Z1 < Z2 (正常情况; 偶尔 Z1≈Z2 说明数据少)
