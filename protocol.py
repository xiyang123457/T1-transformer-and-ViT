"""
W4-D1 — 迁移协议 run_protocol(cfg): Z0/Z1/Z2 统一入口 + 四口径指标 + 断言

三档本质 (口径 8 / 任务书知识点 3):
    Z0 冻结+NCM : 无梯度提特征, 最近类均值分类 -> "预训练特征本身好不好"
    Z1 只训头   : 冻结骨干, 只训线性头         -> "学一个线性映射够不够"
    Z2 全参微调 : 解冻全量 (分层 lr)           -> "要不要改造特征本身"
输入: `ProtocolCfg`
输出: 一个 dict (身份 9 列 + 四口径 + gap + 资源 + 卫生 + history), 供 run_matrix 落盘

跑完应看到的自检数字 (见文件末尾「怎么验证跑对了」):
    - Z0: backbone 权重零变化 + `ncm_check.refit_acc >= 0.99` + `|ncm - knn1| <= 0.02`
    - Z1: 只有 head 权重变化 + backbone 全部 requires_grad=False
    - Z2: 至少一层 backbone 权重变化
    - 三个架构的 `top1_val` 都应明显高于随机 (1/102 ≈ 0.98%)
"""
# ==== 1. 依赖导入 ====
import copy                                             # 深拷贝最优权重
import os                                               # 环境变量 (CUBLAS_WORKSPACE_CONFIG)
import random                                           # Python 内置随机源 (torchvision 增强用它)
import time                                             # 计时 (minutes / 单 epoch 秒数)
import numpy as np                                      # 指标统计 / 种子
import torch                                            # 张量 / 训练 / AMP
import torch.nn as nn                                   # CrossEntropyLoss / Linear
from torch.utils.data import DataLoader, Subset          # 批加载 / 子集

from config import ProtocolCfg, SUBSET_SEED              # 唯一配置源 + 子集种子
from data import (build_transform, get_dataset, get_raw_dataset,   # 数据层
                  get_targets, load_tier_indices)

# ==== 2. 确定性开关 (D1 自检要求: 同组重跑"数字完全一致") ====
def set_determinism():
    """
    关掉 CUDA 上的非确定性来源

    为什么必须做:
        cuDNN 的卷积/反传默认可能用 atomicAdd 等非确定实现, 同一份输入两次跑会差 ~1e-6,
        训练二十轮后被放大 -> 实测出现过 top1 差 2.4 个点 (0.4382 vs 0.4618)。
        对"补种子算 σ"的实验, 这种噪声会直接污染误差棒, 必须在源头掐掉。
    """
    # CUBLAS_WORKSPACE_CONFIG: cuBLAS 走确定性路径所需的 workspace 配置 (必须在首次 CUDA 调用前设置)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True      # 卷积用确定性算法
    torch.backends.cudnn.benchmark = False         # 关自动调优 (调优结果会随机器状态变)
    # 把 SDPA 强制到 math 后端: flash / mem-efficient 的【反向】是非确定性实现
    #   实测告警: "Memory Efficient attention defaults to a non-deterministic algorithm"
    #   -> ViT/DeiT 同配置重跑会漂 ~0.6 个点; math 后端反向是确定的
    for _fn, _flag in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False),
                       ("enable_cudnn_sdp", False), ("enable_math_sdp", True)):
        if hasattr(torch.backends.cuda, _fn):        # 不同 torch 版本函数名可能缺
            getattr(torch.backends.cuda, _fn)(_flag)
    # use_deterministic_algorithms(True, warn_only=True):
    #   尽量强制确定性; 个别在 CUDA 上无确定性实现的算子只告警不中断 (否则整个训练跑不起来)
    torch.use_deterministic_algorithms(True, warn_only=True)

# ==== 3. 模型构建 (按注册表; 口径 1/4) ====
def build_model(cfg):
    """
    按 cfg.spec 建模型并把分类头换成 cfg.num_classes

    参数:
        cfg (ProtocolCfg)
    返回:
        nn.Module
    """
    spec = cfg.spec
    if spec.library == "torchvision":
        import torchvision.models as models
        # models.ResNet50_Weights[tag] -> 权重枚举成员 (tag 例 "IMAGENET1K_V2")
        #   作用: 取官方预训练权重对象, 传给 weights= 参数
        #   坑: 枚举取值用 [] 而不是属性名, 否则大小写不符会 AttributeError
        weights = models.ResNet50_Weights[spec.tag]
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)    # 2048 -> 102
        return model
    import timm
    model_name = f"{spec.model_id}.{spec.tag}"        # 例 vit_small_patch16_224.augreg_in1k
    # timm.create_model(name, pretrained=True, num_classes=102) -> nn.Module
    #   作用: 建结构 + 加载指定 tag 的预训练权重 + 换头
    #   关键参数: num_classes 对 DeiT 蒸馏版会同时改 head 与 head_dist (口径 1: 两者都要 102)
    return timm.create_model(model_name, pretrained=True, num_classes=cfg.num_classes)

def head_module(model):
    """取分类头模块 (torchvision ResNet 叫 fc, timm ViT/DeiT 叫 head)"""
    return model.fc if hasattr(model, "fc") else model.head

def head_params(model):
    """取分类头参数列表"""
    return list(head_module(model).parameters())

# ==== 4. 统一前向 (口径 2/3: 训练与评估走同一条路, 且永不碰两头平均) ====
def forward_logits(model, x, spec):
    """
    统一前向 -> 分类 logits (B, num_classes)

    参数:
        model (nn.Module); x (Tensor (B,C,H,W)); spec (ArchSpec)
    返回:
        Tensor (B, num_classes)
    """
    # 设计决策: timm 架构显式走 "[CLS] -> final norm -> head", 绝不调 model(x)
    #   实测依据 (timm 1.0.30, 2026-10-06 实机核对):
    #     ① 蒸馏版 eval 下 model(x) 既不等于 head(norm(CLS)), 也不等于两头 logits 平均 -> 不可用
    #     ② ⚠ 更隐蔽的一条: 蒸馏版的 forward_head(feat, pre_logits=True) 返回的是
    #        **(CLS+dist)/2 的 token 均值**(实测 max|diff| = 0.0), 并不是 norm(CLS)。
    #        它绕开了"两头 logits 平均", 却撞上"token 平均" -> 喂给 head 的仍是混合特征,
    #        dist_token 会被 task loss 更新 (W4 无 teacher, 它本该一动不动), 与口径 2 不符
    #     ③ 非蒸馏版 (ViT-S) 的 pre_logits 恰好 = norm(CLS) (差 2.6e-06, 数值噪声)
    #   -> 统一写 norm(feat[:, 0]): 两版都正确, 且 eval_token='cls' 的记录才与实现一致
    if spec.library == "timm":
        feat = model.forward_features(x)      # (B, N(+1), D)  蒸馏版 N 多一个 dist token
        pre = model.norm(feat[:, 0])          # (B, D) [CLS] 过 final LayerNorm
        return model.head(pre)                # (B, C)
    return model(x)                           # torchvision ResNet (无 token 概念)

def extract_features(model, loader, device, spec):
    """
    冻结前向, 截获分类头的输入 = 池化后的特征 (B, D)

    参数:
        model, loader, device, spec
    返回:
        (feats (N,D) CPU tensor, labels (N,) CPU tensor)
    """
    # register_forward_hook -> RemovableHandle
    #   作用: 在 head 前向时触发, inp[0] 就是 head 的输入 (即池化特征)
    #   坑: 用完必须 remove(), 否则 hook 残留会污染后续所有前向
    head = head_module(model)
    box = {}
    handle = head.register_forward_hook(
        lambda m, inp, out: box.__setitem__("f", inp[0].detach()))
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            forward_logits(model, x.to(device), spec)     # 与训练同一条前向 -> hook 自然触发
            feats.append(box["f"].cpu())
            labels.append(y)
    handle.remove()
    return torch.cat(feats), torch.cat(labels)

# ==== 5. 四口径指标 (口径 8; val 上 top1 恒等于 recall_mean) ====
def classification_metrics(preds, labels, num_classes):
    """
    算四个核心指标

    参数:
        preds: 预测类别 (序列); labels: 真类别 (序列); num_classes (int)
    返回:
        dict — top1_val / macro_f1_val / recall_mean_val / recall_min5_val
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    top1 = float((preds == labels).mean())
    # 变量 recalls: list[float], 有样本的类各自的召回
    #   为什么只看 support>0 的类: 没有样本的类召回无定义, 计入会人为拉低 balanced acc
    recalls, f1s = [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        support = int((labels == c).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)                                # macro-F1 按 102 类等权平均
        if support > 0:
            recalls.append(recall)
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    worst5 = sorted(recalls)[:5]                      # 最差 5 类
    return {
        "top1_val": top1,
        "macro_f1_val": float(np.mean(f1s)),
        "recall_mean_val": recall_mean,
        "recall_min5_val": float(np.mean(worst5)) if worst5 else 0.0,
    }

@torch.no_grad()
def evaluate(model, loader, device, spec, num_classes, criterion=None):
    """
    在 loader 上评估 (含可选 val_loss)

    参数:
        model, loader, device, spec, num_classes
        criterion: 传了就同时算 loss
    返回:
        (metrics dict, val_loss float|None, preds np.ndarray, labels np.ndarray)
    """
    # model.eval(): 关 dropout / BN 用滑动统计 (忘调会让评估不稳定)
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = forward_logits(model, x, spec)                 # (B, C)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.numel()  # 按样本数加权
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    val_loss = loss_sum / len(loader.dataset) if criterion is not None else None
    return classification_metrics(preds, labels, num_classes), val_loss, preds, labels

# ==== 6. Z0: 冻结 + NCM (含回代与 kNN 互检) ====
def run_Z0(model, tr_loader, va_loader, device, cfg):
    """
    零训练协议: 提特征 -> 类中心 -> 最近类均值分类 (+ 回代 + kNN(1) 互检)

    参数:
        model, tr_loader, va_loader, device, cfg
    返回:
        (metrics dict, ncm_check dict, preds np.ndarray, labels np.ndarray)
    """
    spec = cfg.spec
    f_tr, y_tr = extract_features(model, tr_loader, device, spec)    # (N_tr, D)
    f_va, y_va = extract_features(model, va_loader, device, spec)    # (N_va, D)
    # 逐行 L2 归一化: 让距离只比方向不比模长 (NCM 常规做法)
    f_tr = f_tr / f_tr.norm(dim=1, keepdim=True)
    f_va = f_va / f_va.norm(dim=1, keepdim=True)
    classes = torch.unique(y_tr)                                     # (102,)
    # 每类特征均值 = "类中心" -> (102, D)
    means = torch.stack([f_tr[y_tr == c].mean(0) for c in classes])
    # 校验 1: 类中心 shape 必须是 (102, D) 且 D 对得上该架构
    dist = torch.cdist(f_va, means)                                  # (N_va, 102)
    pred = classes[dist.argmin(1)].numpy()
    metrics = classification_metrics(pred, y_va.numpy(), cfg.num_classes)
    # 校验 2: 回代准确率 (用训练集自身去分) —— 正常应 >= 0.99
    dist_tr = torch.cdist(f_tr, means)
    pred_tr = classes[dist_tr.argmin(1)].numpy()
    refit_acc = float((pred_tr == y_tr.numpy()).mean())
    # 校验 3: kNN(k=1) 与 NCM 互检 —— 两者应差 <= 1~2 个点
    knn_pred = y_tr[torch.cdist(f_va, f_tr).argmin(1)].numpy()
    knn1_acc = float((knn_pred == y_va.numpy()).mean())
    ncm_check = {
        "center_shape": list(means.shape),
        "feat_dim": int(f_tr.shape[1]),
        "refit_acc": round(refit_acc, 6),
        "knn1_acc": round(knn1_acc, 6),
        "gap_ncm_knn1": round(abs(metrics["top1_val"] - knn1_acc), 6),
        "ok": bool(refit_acc >= 0.99 and abs(metrics["top1_val"] - knn1_acc) <= 0.02),
    }
    return metrics, ncm_check, pred, y_va.numpy()

# ==== 7. 训练/评估循环 (Z1/Z2 共用; AMP + 逐轮 history) ====
def train_and_eval(model, tr_loader, va_loader, optimizer, device, cfg):
    """
    训练 + 早停; 返回最优权重与逐轮 history

    参数:
        model, tr_loader, va_loader, optimizer, device, cfg
    返回:
        (best_state dict, best_epoch int, epochs_run int, history list[dict])
    """
    spec = cfg.spec
    criterion = nn.CrossEntropyLoss()
    amp = torch.cuda.is_available()                    # 只有 CUDA 才开混合精度
    # 变量 scaler: GradScaler, 混合精度的梯度缩放器
    #   为什么: fp16 下小梯度会下溢成 0 -> 放大再反传; enabled=False 时全部直通
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_top1, best_epoch, best_state, bad = -1.0, -1, None, 0
    history, epochs_run = [], 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # torch.amp.autocast("cuda", enabled=amp): 前向用 fp16 加速 (数值敏感处自动回退 fp32)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = forward_logits(model, x, spec)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()              # 缩放后反传
            scaler.step(optimizer)                     # 更新 (含 inf/nan 跳过)
            scaler.update()
            running += loss.item() * y.numel()
        tr_loss = running / len(tr_loader.dataset)
        metrics, va_loss, _, _ = evaluate(model, va_loader, device, spec,
                                          cfg.num_classes, criterion)
        top1 = metrics["top1_val"]
        lr = optimizer.param_groups[0]["lr"]           # 记录 lr -> 证明 schedule 真的在跑
        epochs_run = epoch + 1
        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "val_loss": round(float(va_loss), 6), "val_top1": round(top1, 6),
            "lr": lr, "seconds": round(time.time() - t0, 2),
        })
        print(f"  [{epoch:02d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_top1={top1:.4f} lr={lr:.2e} ({history[-1]['seconds']}s)")
        if top1 > best_top1:
            best_top1, best_epoch = top1, epoch
            best_state = copy.deepcopy(model.state_dict())   # 快照"最优那一刻"的权重
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stop @ epoch {epoch} (best {best_top1:.4f} @ {best_epoch})")
                break
    return best_state, best_epoch, epochs_run, history

# ==== 8. Z1 / Z2 ====
def run_Z1(model, tr_loader, va_loader, device, cfg):
    """Z1: 冻结骨干, 只训头 (线性探针)"""
    for p in model.parameters():
        p.requires_grad = False                        # 先全冻
    hp = head_params(model)
    for p in hp:
        p.requires_grad = True                         # 只解冻头
    # AdamW(头参数, lr_head): Z1 只用一个大 lr
    opt = torch.optim.AdamW(hp, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    return train_and_eval(model, tr_loader, va_loader, opt, device, cfg)

def run_Z2(model, tr_loader, va_loader, device, cfg):
    """Z2: 全参微调 (骨干小 lr + 头大 lr)"""
    for p in model.parameters():
        p.requires_grad = True                         # 全解冻
    hp = head_params(model)
    hids = {id(p) for p in hp}                         # 用 id 判重, 保证两组互补不重叠
    bb = [p for p in model.parameters() if id(p) not in hids]
    # AdamW(param_groups): 两组不同 lr; 只作用训练集, 不影响其它
    opt = torch.optim.AdamW([
        {"params": bb, "lr": cfg.lr_backbone},
        {"params": hp, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    return train_and_eval(model, tr_loader, va_loader, opt, device, cfg)

# ==== 9. 断言 (口径 A5: 跑完立刻验, 不做事后补) ====
def check_assertions(cfg, model, before, metrics):
    """
    按档位生成断言结果

    参数:
        cfg; model (训练后的模型); before (训练前 {名字: 权重快照}); metrics (val 四口径)
    返回:
        dict — {mode, changed_head, changed_backbone, backbone_frozen, head_classes_ok,
                val_acc_sane, ok}
    """
    head_ids = {id(p) for p in head_params(model)}
    changed_head, changed_bb = [], []
    for name, p in model.named_parameters():
        changed = (name not in before) or (not torch.equal(before[name], p.detach()))
        if not changed:
            continue
        (changed_head if id(p) in head_ids else changed_bb).append(name)
    backbone_frozen = all(not p.requires_grad for n, p in model.named_parameters()
                          if id(p) not in head_ids)
    # head_classes_ok: 头输出维必须 = 102; DeiT 蒸馏版的 head_dist 也要 102 且评估不走它
    head = head_module(model)
    head_classes_ok = bool(getattr(head, "out_features", -1) == cfg.num_classes)
    hd = getattr(model, "head_dist", None)
    if hd is not None:
        head_classes_ok = head_classes_ok and getattr(hd, "out_features", -1) == cfg.num_classes
    # val_acc_sane: 口径 A5 只对 Z2 提这个要求 (Z0/Z1 不判)
    #   为什么: Z1 只训一个线性头, 短训/小数据时分数低是正常的, 不能当成"口径错"
    val_acc_sane = bool(metrics["top1_val"] > 0.05) if cfg.mode == "Z2" else True
    if cfg.mode == "Z0":
        ok = (len(changed_head) == 0 and len(changed_bb) == 0)
    elif cfg.mode == "Z1":
        ok = (len(changed_head) > 0 and len(changed_bb) == 0 and backbone_frozen)
    else:
        ok = (len(changed_bb) > 0)
    return {
        "mode": cfg.mode,
        "changed_head": changed_head,
        "changed_backbone": changed_bb,
        "backbone_frozen": backbone_frozen,
        "head_classes_ok": head_classes_ok,
        "val_acc_sane": val_acc_sane,
        "ok": bool(ok and head_classes_ok and val_acc_sane),
    }

# ==== 10. 统一入口 ====
def run_protocol(cfg: ProtocolCfg):
    """
    跑一格完整协议

    参数:
        cfg (ProtocolCfg)
    返回:
        dict — 身份 9 列 + 四口径 + gap + 资源 + 卫生 + history (另含 _preds/_labels 供 run_matrix)
    """
    t_start = time.time()
    set_determinism()          # 先关掉 CUDA 非确定性来源 (必须在任何 CUDA 计算之前)
    # 固定所有随机源 (缺一不可), 保证"同配置重跑数字一致":
    #   random -> torchvision 的 RandomResizedCrop / RandomHorizontalFlip 用的是 **Python random**,
    #             只 seed torch/numpy 时增强会每次不同 -> 实测 top1 差 2.2 个点 (0.3618 vs 0.3843)
    #   numpy  -> 分层抽样 / 指标统计
    #   torch  -> 权重初始化 / DataLoader shuffle / CUDA 算子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()           # 峰值显存从 0 起算
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = cfg.spec

    model = build_model(cfg).to(device)
    print(f"[{cfg.eid}] device={device} | head={head_module(model)}")

    # 变换必须由 model 解析 (口径 3): 记录"真实用的" mean/std, 而不是手抄常数
    # ⚠ 口径 6 的落点: 训练变换必须由 cfg.train_aug 决定, 不能恒传 True ——
    #   Z0 是"提特征 + NCM", 若用带随机裁剪/翻转的变换提特征, 类中心会混入增强噪声,
    #   而 val 走 center crop -> 训练/评估分布不匹配, 整行 Z0 系统性偏低 (实测掉 7~11 点)
    tr_tf, data_cfg = build_transform(spec, cfg.train_aug, model=model)
    ev_tf, _ = build_transform(spec, False, model=model)
    preproc = cfg.preproc_str(data_cfg["mean"], data_cfg["std"])
    tier_idx = _tier_idx(cfg)          # 只取一次: 每次调用都要重扫 train 标签, 批跑时是纯浪费

    train_loader = DataLoader(Subset(get_dataset("train", tr_tf), tier_idx),
                              batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers)
    # "关增强"训练集只用于算 gap (清单 A4); Z0 无训练 -> 不构建 (省一次数据集扫描)
    train_eval_loader = None
    if cfg.mode != "Z0":
        train_eval_loader = DataLoader(Subset(get_dataset("train", ev_tf), tier_idx),
                                       batch_size=cfg.batch, shuffle=False,
                                       num_workers=cfg.num_workers)
    val_loader = DataLoader(get_dataset("val", ev_tf), batch_size=cfg.batch,
                            shuffle=False, num_workers=cfg.num_workers)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ncm_check, history, best_epoch, epochs_run = None, [], -1, 0
    preds = labels = None

    if cfg.mode == "Z0":
        metrics, ncm_check, preds, labels = run_Z0(model, train_loader, val_loader, device, cfg)
    else:
        runner = run_Z1 if cfg.mode == "Z1" else run_Z2
        best_state, best_epoch, epochs_run, history = runner(
            model, train_loader, val_loader, device, cfg)
        if best_state is not None:
            model.load_state_dict(best_state)          # 回到最优权重再算最终指标
        metrics, _, preds, labels = evaluate(model, val_loader, device, spec, cfg.num_classes)

    if cfg.mode != "Z0":
        # top1_train: 必须用"关增强"的确定性变换, 否则会低估 gap (清单 A4)
        tr_metrics, _, _, _ = evaluate(model, train_eval_loader, device, spec, cfg.num_classes)
        top1_train = tr_metrics["top1_val"]
        gap = top1_train - metrics["top1_val"]
    else:
        top1_train, gap = None, None

    checks = check_assertions(cfg, model, before, metrics)
    minutes = (time.time() - t_start) / 60.0
    peak_mem = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0

    result = {
        # —— 身份 (date / git_commit / json_path 由 run_matrix 补) ——
        "eid": cfg.eid, "arch": cfg.arch, "mode": cfg.mode.upper(), "data": cfg.data,
        "aug": cfg.aug, "distill": int(cfg.distill), "seed": cfg.seed,
        "eval_token": cfg.eval_token, "preproc": preproc,
        # —— 训练过程 ——
        "best_epoch": best_epoch, "epochs_run": epochs_run,
        # —— 四口径 ——
        **metrics,
        # —— 过拟合 (A4) ——
        "top1_train": top1_train, "gap": gap,
        # —— 资源 (A6) ——
        "minutes": round(minutes, 3), "peak_mem_gb": round(peak_mem, 3), "degraded": cfg.degraded,
        # —— 卫生 (A5) ——
        "assert_ok": bool(checks["ok"]), "ncm_check": ncm_check,
        # —— 明细 (不进 CSV, 进 JSON) ——
        "checks": checks, "history": history, "data_cfg": _jsonable(data_cfg),
        "config": {k: v for k, v in cfg.__dict__.items()},
        # —— 私有: run_matrix 用于 --save-detail, 不写进 JSON ——
        "_preds": preds, "_labels": labels,
    }
    print(f"[{cfg.eid}] top1_val={metrics['top1_val']:.4f} macro_f1={metrics['macro_f1_val']:.4f} "
          f"recall_min5={metrics['recall_min5_val']:.4f} assert_ok={result['assert_ok']} "
          f"({minutes:.2f}min)")
    return result

def _tier_idx(cfg):
    """
    取该格的 train 子集索引 (含 hash / 种子 / 漂移 三级校验)

    参数:
        cfg (ProtocolCfg)
    返回:
        np.ndarray[int64]
    """
    # 注意: 子集种子用 SUBSET_SEED(42), 而不是 cfg.seed —— 补种子时子集必须不变
    targets = get_targets(get_raw_dataset("train"))
    idx, _rec = load_tier_indices(cfg.data, SUBSET_SEED, targets)
    return idx

def _jsonable(obj):
    """把 numpy/tuple 等转成可 JSON 序列化的形式"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ==== 怎么验证跑对了 ====
# 1) Z0: `result["checks"]["changed_backbone"] == []` 且 `ncm_check["refit_acc"] >= 0.99`
#    且 `ncm_check["center_shape"] == [102, D]` (D: resnet50 2048 / vit_small 384 / deit_tiny 192)
# 2) Z1: `checks["changed_head"]` 非空、`changed_backbone == []`、`backbone_frozen is True`
# 3) Z2: `checks["changed_backbone"]` 非空
# 4) 三个架构 `top1_val` 都远高于 1% (随机水平); 否则说明标签/口径错了
# 5) 同配置连跑两次, 四口径数字应完全一致 (种子固定 + 子集 hash 一致)
