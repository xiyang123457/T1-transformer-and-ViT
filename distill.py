"""
distill.py — 手写 hard 蒸馏 (W5 维度 D 的核心装置)

用途:
    把「要不要 teacher」做成一个可插拔开关, 供 protocol.train_and_eval 调用。只做四件事:
      ① 取 DeiT 蒸馏版的**两路 logits** (口径 24: 主口径走 cls, 蒸馏项走 dist)
      ② 把 teacher (ResNet-50) 包成**在线前向**的硬标签来源 (口径 27)
      ③ 手写 hard 蒸馏损失 (1-α)·CE(cls,y) + α·CE(dist,y_t)   (口径 26: α=0.5, 无温度 T)
      ④ batch 级 Mixup/CutMix 挂载 (口径 25: DeiT 配方, 故意不含 reprob / drop-path)
输入:
    checkpoints/teacher_resnet50.pt   (W3 产物; 顶层键 model / config / val_top1 / epoch)
    学生的两路 logits、目标标签、增强且 mixup 之后的 batch
输出:
    标量损失 (+ 分量 dict); 或副读数 logits (cls+dist)/2

自检 (python distill.py):
    [align] cls 通路          max|diff| <= 1e-5   (与 timm 官方前向的等价性)
    [align] dist 通路         max|diff| == 0.0
    [align] 蒸馏损失          max|diff| <= 1e-6   (手写 vs 展开式)
    [align] α=0 / α=1 极端    max|diff| <= 1e-6   (退化性: 纯 CE / 纯蒸馏)
    [align] 软标签 CE         max|diff| <= 1e-6   (手写 vs timm.loss.SoftTargetCrossEntropy)
    [align] teacher val top1  == 0.9333           (类别顺序未错位 —— 唯一能当场抓此类 bug 的检查)
    [align] y_t != y 不一致率 basic 1-2.4% / strong 15-17%   (口径 28 的预登记依据)

怎么验证跑对了:
    1) `python distill.py` -> 打印 [align] 表格, 末行 `[align] PASS`, 退出码 0
    2) `teacher val top1` 必须是 0.9333 (与 W3 记录一致); 差很多 = 类别顺序或权重加载错
    3) `y_t != y (basic)` 应 ≈ 0.01-0.024: 若 ≈ 0 说明 teacher 与 student 看到的是同一批干净图
       (即"在线前向"没接上, 蒸馏会退化成第二个 CE —— 口径 27 要防的正是这个)
"""
# ==== 1. 依赖导入 ====
import torch                         # 张量
import torch.nn as nn                # nn.Module 基类 (TeacherWrapper 要继承)
import torch.nn.functional as F      # cross_entropy / log_softmax / one_hot

# 变量 TEACHER_CKPT: str, teacher 检查点绝对路径 (W3-D2 产物)
#   示例值: r"d:/learning project/T1 transformer and ViT/checkpoints/teacher_resnet50.pt"
#   为什么用绝对路径: 批跑可能从任意 CWD 启动 (W4 已踩过相对路径的坑)
TEACHER_CKPT = r"d:/learning project/T1 transformer and ViT/checkpoints/teacher_resnet50.pt"

# 变量 NUM_CLASSES: int, 类别数 (Flowers102)
#   示例值: 102
#   为什么单列一个常量: timm.data.Mixup 的 num_classes 默认 1000, 必须显式传 102
#     否则 one-hot 宽度错且**不报错** (静默算错 —— 最难查的一类)
NUM_CLASSES = 102

# ==== 2. 两路 logits (口径 24) ====
def two_path_logits(model, x):
    """
    取 DeiT 蒸馏版的两路 logits (主口径 cls + 蒸馏项 dist)

    参数:
        model (nn.Module): deit_tiny_distilled_patch16_224 (head / head_dist 已换成 102 类)
        x (Tensor): (B, 3, 224, 224) 已归一化
    返回:
        (logits_cls (B,102), logits_dist (B,102))
    """
    # 设计决策: 显式走 "token -> final norm -> head", 绝不走下面两条捷径 ——
    #   ① model(x): 蒸馏版 eval 下返回 (head(cls)+head_dist(dist))/2, 与口径 24 直接冲突
    #   ② forward_head(feat, pre_logits=True): 实测返回 (CLS+dist)/2 的 **token 均值**
    #      (max|diff| = 0.0), 不是 norm(CLS); 而且会让 dist_token 被 task loss 更新
    #   (两条都是 W4 实机核对过的, 见 protocol.py:101-109, 不是理论洁癖)
    if not (hasattr(model, "head_dist") and hasattr(model, "dist_token")):
        raise AttributeError("two_path_logits 只适用于蒸馏版 DeiT (需 dist_token + head_dist)")
    feat = model.forward_features(x)      # (B, 198, 192): 0=cls, 1=dist, 2..=patch
    pre_cls = model.norm(feat[:, 0])      # (B, 192) [CLS] 过 final LayerNorm
    pre_dist = model.norm(feat[:, 1])     # (B, 192) [dist] 过**同一个** LayerNorm
    return model.head(pre_cls), model.head_dist(pre_dist)

# ==== 3. TeacherWrapper (口径 27: 在线前向) ====
class TeacherWrapper(nn.Module):
    """
    teacher 包装器: 加载 W3 的 ResNet-50 检查点, 只做前向、不建计算图

    参数:
        ckpt_path (str)  : 检查点路径 (顶层键 model / config / val_top1 / epoch)
        num_classes (int): 期望类别数 (102)
        device (str)     : "cuda" / "cpu"
    提供:
        forward(x)    -> logits (B,102)
        hard_label(x) -> (B,) long: teacher 对**同一份输入**的 argmax 硬标签
    """

    def __init__(self, ckpt_path=TEACHER_CKPT, num_classes=NUM_CLASSES, device="cuda"):
        super().__init__()                     # 建立 nn.Module 内部机制 (参数登记等)
        import torchvision.models as tvm       # 延迟导入: 只在真要蒸馏时才付这段导入开销
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config") or {}
        # —— 口径 5 的两条断言: "类别顺序错位"是最难查的一类 bug ——
        #   症状: hard label 全指错类, 而 loss 照样下降 (看起来"训练正常")
        #   当场能抓住它的只有这两条 + 自检里的 val 复算
        assert cfg.get("num_classes") == num_classes, \
            f"teacher num_classes={cfg.get('num_classes')} != {num_classes}"
        classes = cfg.get("classes")
        assert classes is not None and len(classes) == num_classes, \
            f"teacher classes 长度={len(classes) if classes else None} != {num_classes}"

        net = tvm.resnet50(num_classes=num_classes)   # 结构必须与训练 teacher 时完全一致
        net.load_state_dict(ckpt["model"])            # 顶层 'model' = 320 键的 state_dict
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)          # 关键: teacher 不参与反传 -> 省显存与时间 (口径 27 的前提)
        self.net = net.to(device)
        self.device = device
        self.num_classes = num_classes
        # 变量 teacher_mean/std: tuple[float, ...], teacher 训练时的归一化常数
        #   示例值: (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)
        #   为什么存下来: 口径 27 要求 teacher 与 student 吃**同一份** batch; 若两者归一化不同,
        #     "同一份 batch" 就不成立, 必须先断言再跑 (见 assert_teacher_preproc_matches)
        self.teacher_mean = tuple(cfg.get("imagenet_mean", (0.485, 0.456, 0.406)))
        self.teacher_std = tuple(cfg.get("imagenet_std", (0.229, 0.224, 0.225)))
        self.val_top1 = float(ckpt.get("val_top1", float("nan")))

    @torch.no_grad()                          # 双保险: 除 requires_grad_(False) 外再关梯度
    def forward(self, x):
        return self.net(x)                    # (B, 102) logits

    @torch.no_grad()
    def hard_label(self, x):
        return self.forward(x).argmax(dim=1)  # (B,) long —— 这就是 DeiT 的 hard 蒸馏信号


def assert_teacher_preproc_matches(teacher, data_cfg, atol=1e-6):
    """
    口径 27 的前置断言: teacher 与 student 必须用同一套归一化

    参数:
        teacher (TeacherWrapper): 已加载的 teacher
        data_cfg (dict): student 的 timm 数据配置 (含 mean/std)
        atol (float): 容差
    返回:
        None (不满足则抛 AssertionError)
    """
    # 为什么必须有这一条: 在线前向的语义是"teacher 与 student 看同一张张量"。
    #   若归一化不同, 同一张张量对 teacher 就是"分布错的输入" -> hard label 系统性变差,
    #   而 Δ蒸馏 会被误读成"蒸馏没用"。本项目 DeiT(fb_in1k) 与 ResNet-50(IMAGENET1K_V2)
    #   恰好都是 ImageNet-1k 常数, 所以可以直接共用 —— 但这件事必须被断言, 不能靠"恰好"
    tm, ts = teacher.teacher_mean, teacher.teacher_std
    dm = tuple(round(float(v), 6) for v in data_cfg["mean"])
    ds = tuple(round(float(v), 6) for v in data_cfg["std"])
    assert all(abs(a - b) <= atol for a, b in zip(tm, dm)), f"mean 不一致: teacher={tm} student={dm}"
    assert all(abs(a - b) <= atol for a, b in zip(ts, ds)), f"std 不一致: teacher={ts} student={ds}"

# ==== 4. 手写 hard 蒸馏损失 (口径 26) ====
def soft_target_cross_entropy(logits, target):
    """
    软标签交叉熵 (= timm.loss.SoftTargetCrossEntropy 的同式实现)

    参数:
        logits (Tensor): (B, C) 未归一化分数
        target (Tensor): (B, C) 软标签 (行和 == 1)
    返回:
        标量 Tensor
    """
    # 为什么自己写而不是直接 import: mixup 激活时 base loss 必须用它 (坑位 2);
    #   而"手写 + 与 timm 官方对齐"正是验收 3 要证明的东西
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def hard_distill_loss(logits_cls, logits_dist, target, y_teacher, alpha=0.5):
    """
    hard 蒸馏损失: (1-α)·CE(cls, y) + α·CE(dist, y_t)

    参数:
        logits_cls (Tensor): (B,102) 主口径 logits (head 那一路)
        logits_dist (Tensor): (B,102) 蒸馏路 logits (head_dist 那一路)
        target: 硬标签 (B,) long; 或 mixup 后的软标签 (B,102) float
        y_teacher (Tensor): (B,) long, teacher 对**同一份 batch** 的硬标签
        alpha (float): 蒸馏项权重 (口径 26 定稿 0.5)
    返回:
        (loss Tensor, parts dict) — parts 供落盘/排查
    """
    # 第 1 步: base 项。mixup 激活时 target 是软标签 -> 必须换 SoftTargetCrossEntropy
    #   坑: 直接 F.cross_entropy(软标签) 会报错或算错 (它期望 long 类号)
    if target.dtype == torch.long:
        base = F.cross_entropy(logits_cls, target)
    else:
        base = soft_target_cross_entropy(logits_cls, target)
    # 第 2 步: 蒸馏项。**纯 CE, 不加 label_smoothing** ——
    #   为什么: y_t 是 argmax 出来的硬类号; 平滑会把它变成"高置信软标签", 偏离 DeiT 的 hard 蒸馏定义
    distill = F.cross_entropy(logits_dist, y_teacher)
    loss = (1.0 - alpha) * base + alpha * distill
    return loss, {"base": float(base.detach()), "distill": float(distill.detach()),
                  "alpha": alpha}

# ==== 5. batch 级 Mixup/CutMix (口径 25) ====
def build_mixup(num_classes=NUM_CLASSES):
    """
    构建 DeiT 配方的 Mixup/CutMix (batch 级)

    参数:
        num_classes (int): 类别数, **必须显式传 102**
    返回:
        timm.data.Mixup 实例 (callable: (x, target_classid) -> (mixed_x, mixed_target_soft))
        其中 target_classid 是 **(B,) int64 类号**, 返回值 mixed_target_soft 是 (B, C) 软标签
    """
    import timm
    # timm.data.Mixup(mixup_alpha, cutmix_alpha, cutmix_minmax, prob, switch_prob, mode,
    #                 label_smoothing, num_classes)
    #   作用  : 每个 batch 以 prob=1.0 的概率施加 Mixup 或 CutMix (switch_prob=0.5 各半)
    #   关键参数: 照 DeiT 官方值 mixup=0.8 / cutmix=1.0 / switch=0.5 / ls=0.1; 报告注明"未自调"
    #   坑 1: num_classes 默认 1000 -> 必须显式传 102, 否则 one-hot 宽度错且不报错
    #   坑 2 (2026-10-07 踩过并修): target 要的是 **类号 (B,) int64**, 不是 one-hot ——
    #     它内部 mixup_target() 会自己 one_hot(y) 再加 label_smoothing。
    #     传 one-hot 进去 = 做 one_hot(one_hot(y)) -> 返回 (B*C, C), 训练时立刻报
    #     "size of tensor a (3264) must match b (32)"; 静态回归检查见 verify_w5d2.py 的 S6
    #   坑 3: 内含 label_smoothing=0.1 -> base loss 只能配 soft_target_cross_entropy
    #   坑 4: 只能在训练循环里调 (batch 级) —— 放进 Dataset.transform 拿不到 label, 也不可能有 cutmix
    return timm.data.Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode="batch",
        label_smoothing=0.1, num_classes=num_classes,
    )

# ==== 6. 副读数 (口径 24) ====
def distavg_logits(logits_cls, logits_dist):
    """副读数: 两头 logits 平均 —— 只进 experiments.csv 的 top1_val_distavg 列, 不影响主口径"""
    return (logits_cls + logits_dist) / 2.0

# ==== 7. 自检: 与 timm 官方数值对齐 (验收 3) ====
def selftest_align(device=None, batch=8, full=True):
    """
    三段数值对齐 (只跑一次, 不参与正式训练)

    参数:
        device (str|None): None -> 有 CUDA 就用
        batch (int): 虚拟 batch 大小
        full (bool): True 时额外做 "teacher val top1 == 0.9333" 与 "y_t != y 不一致率" (需要数据)
    返回:
        dict: 各检查项的实测值 (全部达标才算过)
    """
    import timm
    # 坑 (2026-10-07 批跑前置自检抓到): `import timm` **不会**自动暴露子模块 `timm.loss`
    #   -> 直接写 timm.loss.SoftTargetCrossEntropy() 会 AttributeError: module 'timm' has no attribute 'loss'。
    #   子模块必须显式导入 (只有别处恰好 import 过才会"看起来能用", 属于隐式依赖)
    import timm.loss
    from config import ARCH_REGISTRY
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    out = {}

    spec = ARCH_REGISTRY["deitt"]
    model = timm.create_model(f"{spec.model_id}.{spec.tag}", pretrained=True,
                              num_classes=NUM_CLASSES).to(device).eval()
    x = torch.randn(batch, 3, 224, 224, device=device)
    with torch.no_grad():
        cls_mine, dist_mine = two_path_logits(model, x)
        # ① dist 路: 与"模型内置的同一计算"必须逐位相同 (验证我们没写错 norm/索引)
        feat = model.forward_features(x)
        out["dist 通路 max|diff|"] = (dist_mine - model.head_dist(model.norm(feat[:, 1]))).abs().max().item()
        # ② cls 路: 与口径 24 的**展开式**逐位比对 (head(norm(feat[:, 0])))
        #   ⚠ 这里**故意不调 model(x)** —— 计划 §七 明令禁止:
        #     曾用"eval 下 model(x) 等于两头平均"反解 cls 做交叉验证, 2026-10-07 实测该前提不成立
        #     (下面「诊断」那行实测 max|diff| = 1.96), 会得出"cls 通路错了"的假结论。
        #     所以 model(x) 只作为**诊断量**记录, 不进判定
        out["cls 通路 max|diff|"] = (cls_mine - model.head(model.norm(feat[:, 0]))).abs().max().item()
        # 诊断量 (不参与判定): 作为"禁用 model(x)"的实证
        #   为什么兼容 tuple: 若哪天 distilled_training 被误开(timm 0.6- 的坑), model(x) 会返回 tuple,
        #   这里取第 0 项而不是让整个自检崩掉 —— 该误开的正面拦截在 protocol.py 侧
        mx_out = model(x)
        mx_out = mx_out[0] if isinstance(mx_out, (tuple, list)) else mx_out
        out["诊断: model(x) vs 两路平均"] = (mx_out - (cls_mine + dist_mine) / 2.0).abs().max().item()

        # ③ 蒸馏损失: 手写 vs 展开式 + α 两个极端 + 软标签 CE vs timm
        y = torch.randint(0, NUM_CLASSES, (batch,), device=device)
        teacher = TeacherWrapper(device=device)
        y_t = teacher.hard_label(x)
        loss, _ = hard_distill_loss(cls_mine, dist_mine, y, y_t, alpha=0.5)
        ref = 0.5 * F.cross_entropy(cls_mine, y) + 0.5 * F.cross_entropy(dist_mine, y_t)
        out["蒸馏损失 max|diff|"] = abs(loss.item() - ref.item())
        out["α=0 退化误差"] = abs(hard_distill_loss(cls_mine, dist_mine, y, y_t, 0.0)[0].item()
                                  - F.cross_entropy(cls_mine, y).item())
        out["α=1 退化误差"] = abs(hard_distill_loss(cls_mine, dist_mine, y, y_t, 1.0)[0].item()
                                  - F.cross_entropy(dist_mine, y_t).item())
        onehot = F.one_hot(y, NUM_CLASSES).float()
        out["软标签 CE vs timm"] = abs(soft_target_cross_entropy(cls_mine, onehot).item()
                                       - timm.loss.SoftTargetCrossEntropy()(cls_mine, onehot).item())

    if full:
        # ④ teacher 类别顺序复算 + ⑤ y_t != y 不一致率 (口径 28 的预登记依据)
        import data as D
        from torch.utils.data import DataLoader
        tr_tf, tr_cfg = D.build_transform(ARCH_REGISTRY["deitt"], train=True, model=model, aug="basic")
        assert_teacher_preproc_matches(teacher, tr_cfg)          # 口径 27 的前置条件
        # ⚠ 复算 teacher 准确率必须用 **teacher 自己的 (R50) eval 变换**, 不能用 student(deitt) 的:
        #   两者 Resize/Crop/interpolation 不同 (R50: Resize256+CenterCrop224 双线性;
        #   deitt: timm crop_pct=0.875 + bicubic)。2026-10-07 实测: 拿 deitt 变换评 R50 只有 0.9245,
        #   比 ckpt 里记的 0.9333 低 0.88 点 —— 那是变换不匹配, **不是类别错位**。
        #   本检查的目的("类别顺序未错位")只有用 teacher 自己的变换才对得上账 (错位会掉到 ~1%)
        va_tf, _ = D.build_transform(ARCH_REGISTRY["r50"], train=False)
        dl_val = DataLoader(D.get_dataset("val", va_tf), batch_size=batch)
        correct = total = 0
        for xb, yb in dl_val:
            xb, yb = xb.to(device), yb.to(device)
            correct += (teacher.hard_label(xb) == yb).sum().item()
            total += yb.numel()
        out["teacher val top1"] = correct / total
        # teacher 在增强视图上的 top1 + y_t != y 不一致率 (口径 28 的预登记依据)
        #   为什么两个都存: top1 是"在线前向到底接没接上"的照妖镜 (坑位 1 的分级断言要它);
        #   y_t != y 是 Δ蒸馏 的**信息量上界** —— 不一致率≈0 时硬标签≈真标签, 蒸馏项没东西可学
        for aug in ("basic", "strong"):
            tf, _ = D.build_transform(ARCH_REGISTRY["deitt"], train=True, model=model, aug=aug)
            xb, yb = next(iter(DataLoader(D.get_dataset("train", tf), batch_size=256)))
            xb, yb = xb.to(device), yb.to(device)
            agree = (teacher.hard_label(xb) == yb).float().mean().item()
            out[f"teacher top1 (train+{aug})"] = agree
            out[f"y_t != y ({aug})"] = 1.0 - agree
    return out


if __name__ == "__main__":
    res = selftest_align()
    for k, v in res.items():
        print(f"[align] {k:24s} = {v:.6g}")
    ok = (res["cls 通路 max|diff|"] < 1e-5
          and res["dist 通路 max|diff|"] < 1e-9
          and res["蒸馏损失 max|diff|"] < 1e-6
          and res["α=0 退化误差"] < 1e-6
          and res["α=1 退化误差"] < 1e-6
          and res["软标签 CE vs timm"] < 1e-6
          and (res.get("teacher val top1") is None or abs(res["teacher val top1"] - 0.9333) < 1e-3)
          and (res.get("y_t != y (basic)") is None or res["y_t != y (basic)"] < 0.06)
          # —— 坑位 1 的分级断言 (口径 28) ——
          # 为什么只卡 strong 臂: basic 臂的 Δ蒸馏 预期 ≈0 是**结构性**的 (teacher 自己就是
          #   basic 训练出来的, 在 basic 视图上仍约 95.7% 正确), 对它设阈值会误触发;
          # strong 臂才是"在线前向到底接上了没"的照妖镜 —— 实测 teacher 在 RandAugment 视图上
          #   只有约 86.3%。若这里突然 >=0.92, 说明 teacher 看到的是**没加过增强的同一份图**
          #   (即偷缓存了 clean 图), 蒸馏会退化成第二次 CE -> Δ蒸馏 结构性恒为 0 (口径 27)
          and (res.get("teacher top1 (train+strong)") is None
               or res["teacher top1 (train+strong)"] < 0.92))
    print("[align]", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
