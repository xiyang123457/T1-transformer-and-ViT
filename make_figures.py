"""
make_figures.py — W5 维度 D 的图表生成器 (C1 阶梯图 + C3 消融图 + 两张数据驱动补充图)

用途:
    只读 `experiments.csv` (F2 另读 `results/{eid}_history.csv`), 生成 W5 计划 §五 D4 要求的图。
    ⚠ 硬约定 (计划 §七): **图必须由脚本读表生成 —— 不许手改数字、不许在 Excel 里挪柱子**。
       所以本文件里**没有任何写死的绘制数值**: 每个柱高/折点都来自 CSV 单元格;
       文末 `GOLDEN` 里那几个数只用于**断言**(防 CSV 被改坏 / 过滤条件写错), 不参与绘图。
    图内一律**英文标签**: 与 `make_report.py` / `plot_utils.py` 同约定 —— 报告正文中文、图注英文,
       避免 matplotlib 缺 CJK 字体时中文变方框 (W4 已踩过)。

产出 (figures/, 目录不存在会自动创建; 每张图同时出 .png + .pdf, 300 dpi):
    - F1_ladder.png/.pdf            C1: 3 个 subplot (top-1 / macro-F1 / worst-5 recall) —— 计划 D4 规格
    - F1_ladder_<i>_<metric>.png/.pdf  同一数据按 W4 约定"每指标一张 PNG"再出一份 (独立指标不共画布)
    - F3_ablation.png/.pdf          C3: 100% 与 25% 各一子图, 每子图 4 根柱 (基线/强增强/蒸馏/双开),
                                    柱顶标 Δ蒸馏 与 Δ增强; 贴顶组 (best_epoch>=28) 用斜纹标出 (口径 29)
    - F2_curves.png/.pdf            补充: 8 个臂的训练曲线 (loss 与 top-1 配对, 两档各一对)
    - F4_distill_token.png/.pdf     补充: 蒸馏两臂的 主口径(cls) vs 副读数(cls+dist_avg) (口径 24)

输入:
    experiments.csv          42 数据行 / 27 列; 数值列已是"百分数 2 位小数", 空串 = 无该读数
    results/*_history.csv    F2 用 (逐 epoch 曲线; 只有训练组有, Z0 没有)

输出:
    figures/ 下 8 张图 × 2 格式 = 16 个文件

跑完应看到的自检数字 (2026-10-07, 42 组齐时):
    - 终端 `[fig] rows = 42`; 16 个文件; 末行 `[check] PASS`, 退出码 0
    - C3 的 25% 档四根柱 (左→右) = 66.27 / 62.65 / 65.98 / 65.10
    - C3 的 100% 档四根柱 = 92.45 / 92.65 / 92.65 / 91.57
    - C3 里 `deitt-z2-25-s42-strong-distill` 那根柱带斜纹 (= 贴顶 29/30, Δ 读作下界, 口径 29)
    - C1 里 `vits-Z0` 三档 = 95.39 / 97.94 / 98.33 (单调上升, 与 W4 报告表 2 一致)
    - F2 里强增强臂的 val top-1 起点明显低于基线臂 (强增强收敛慢, 坑位 9)

怎么验证跑对了:
    1) `python make_figures.py` -> 打印 16 行 `[fig] ...` 且末行 `[check] PASS` (退出码 0)
    2) 打开 `figures/F3_ablation.png`: 25% 子图四根柱应与上面"自检数字"逐个一致
    3) 打开 `figures/F1_ladder.png`: 三条实线整体上行; 两条虚线 (vits1k) 始终最低
    4) [check] 每次重跑都重新对账 —— 谁改了 CSV 或画错线, 当场 FAIL (而不是靠肉眼扫图)
"""
# ==== 1. 依赖导入 ====
import csv                          # 读 experiments.csv (图的数据来源, 唯一真值)
import os                           # 建目录 / 拼路径 / 报文件大小
import statistics                   # 多 seed 时的均值与标准差 (不引 numpy, 与 make_report.py 一致)

import matplotlib                   # 画图
matplotlib.use("Agg")               # 无界面后端: 只存文件不弹窗 (后台/无人值守安全)
import matplotlib.pyplot as plt     # noqa: E402

# ==== 2. 常量与数据契约 ====
# 变量 CSV_PATH: str, 唯一数据来源
#   示例值: "experiments.csv"
#   为什么不让命令行改: 计划 §七 要求图与汇总表同源; 可改路径 = 可能画到别的表上去
CSV_PATH = "experiments.csv"

# 变量 RESULTS_DIR: str, 逐 epoch 曲线所在目录 (仅 F2 用)
RESULTS_DIR = "results"

# 变量 OUT_DIR: str, 图输出目录 (计划 D4 指定的 figures/)
OUT_DIR = "figures"

# 变量 DPI: int, 出图分辨率 (计划 D4 要求 300 dpi)
DPI = 300

# 变量 TAG: str, 图标题前缀
TAG = "W5"

# 变量 TIERS: tuple[int, ...], C1 阶梯图的 x 轴档位 (口径 11)
TIERS = (25, 50, 100)

# 变量 ABL_TIERS: tuple[int, ...], C3 两个子图的顺序 (先大档后小档, 与计划 §六 表一致)
ABL_TIERS = (100, 25)

# 变量 MODE_COLORS: dict[str, str], 策略配色 (照抄 make_report.py, 保证 W4/W5 同色可对读)
MODE_COLORS = {"Z0": "tab:blue", "Z1": "tab:orange", "Z2": "tab:green"}

# 变量 DASH_ARCHS: tuple[str, ...], 画虚线的架构 (起点消融; W4 约定)
DASH_ARCHS = ("vits1k",)

# 变量 ARM_KEYS: tuple[tuple], 维度 D 的 4 个 arm = (短名, aug, distill)
#   为什么用 aug/distill 而非 eid 后缀来选: 选的是**语义**(翻哪个开关), eid 只是它的编码
ARM_KEYS = (("baseline", "basic", "0"), ("strong", "strong", "0"),
            ("distill", "basic", "1"), ("both", "strong", "1"))

# 变量 ARM_LABEL: dict[str, str], 柱/线的图例名 (英文, 见文件头约定)
ARM_LABEL = {"baseline": "basic (no teacher)", "strong": "strong (no teacher)",
             "distill": "basic + teacher", "both": "strong + teacher"}

# 变量 ARM_COLOR: dict[str, str], F2/F4 里按 arm 上色 (C1 是按 strategy 上色, 两套不冲突)
ARM_COLOR = {"baseline": "tab:gray", "strong": "tab:red",
             "distill": "tab:blue", "both": "tab:purple"}

# 变量 LADDER_METRICS: tuple, C1 的四个指标 = (CSV 列名, y 轴标签)
#   为什么含 gap: 它是 H5 的唯一判据 (计划 §六 口径), 不能只在正文里给数不画
LADDER_METRICS = (("top1_val", "val top-1 (%)"),
                  ("macro_f1_val", "val macro-F1 (%)"),
                  ("recall_min5_val", "val worst-5 recall (%)"),
                  ("gap", "overfit gap (pp)"))

# 变量 LADDER3: tuple[str, ...], 计划 D4 明确要的"3 个 subplot" (并排一张)
LADDER3 = ("top1_val", "macro_f1_val", "recall_min5_val")

# 变量 TOP_EPS: int, 口径 29 的"贴顶"阈值 (best_epoch >= 28 即可能欠训练)
TOP_EPS = 28

# 变量 GOLDEN: dict, **断言用**的抽检值 (不是绘制数据源!)
#   为什么要有: 图与表同源会让"画错线/过滤错 arm"变成静默错误 —— 这几条抽检能把它们当场打出来
GOLDEN = {
    ("vits", "Z0", 25): 95.39, ("vits", "Z0", 50): 97.94, ("vits", "Z0", 100): 98.33,
    ("deitt", "Z2", 25, "basic", "0"): 66.27,
    ("deitt", "Z2", 25, "strong", "0"): 62.65,
    ("deitt", "Z2", 25, "basic", "1"): 65.98,
    ("deitt", "Z2", 25, "strong", "1"): 65.10,
    ("deitt", "Z2", 100, "basic", "0"): 92.45,
    ("deitt", "Z2", 100, "strong", "0"): 92.65,
    ("deitt", "Z2", 100, "basic", "1"): 92.65,
    ("deitt", "Z2", 100, "strong", "1"): 91.57,
}

# ==== 3. 读数工具 ====
def fnum(v):
    """
    CSV 单元格 -> float

    参数:
        v (str|None): 单元格原文
    返回:
        float | None: 空串/None 一律返回 None
    """
    # 为什么必须容忍空串: Z0 没有 top1_train/gap, 非蒸馏臂没有 top1_val_distavg,
    #   DictReader 在这些位置给的是 "" —— 直接 float("") 会 ValueError, 整张图就出不来
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def read_rows(path=CSV_PATH):
    """
    读 experiments.csv

    参数:
        path (str): CSV 路径
    返回:
        list[dict]: 每行一个 dict (键 = 27 个列名)
    """
    # encoding="utf-8-sig" -> 自动吃掉 BOM (Windows PowerShell 写出的 CSV 常带)
    #   omit 会导致第一列名变成 "\ufeffeid", 于是 r["eid"] 全部 KeyError
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def select_arm(rows, arch, mode, tier, aug, distill):
    """
    按 (arch, mode, tier, aug, distill) 取唯一一行

    参数:
        rows (list[dict]); arch/mode (str); tier (int); aug/distill (str): CSV 原样值
    返回:
        dict | None: 命中行; 缺则 None
    """
    for r in rows:
        if (r["arch"] == arch and r["mode"] == mode and int(r["data"]) == tier
                and r["aug"] == aug and r["distill"] == distill):
            return r
    return None


def read_history(eid):
    """
    读某组的逐 epoch 曲线

    参数:
        eid (str): 实验编号
    返回:
        dict[str, list[float]] | None: 列名 -> 数值列; 文件不存在返回 None
    """
    path = os.path.join(RESULTS_DIR, f"{eid}_history.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return {k: [float(r[k]) for r in rows] for k in rows[0].keys()}


def agg_baseline(rows, metric):
    """
    把**基线臂** (aug=basic 且 distill=0) 按 (arch, mode, tier) 聚合

    参数:
        rows (list[dict]); metric (str): CSV 列名
    返回:
        dict: {(arch, mode, tier): (mean, std_or_None, n_seed)}
    """
    # 为什么必须过滤掉 W5 的 6 个消融臂: 它们与基线同 (arch, mode, tier),
    #   混进来会把阶梯图的 deitt-Z2 折点平均掉 (画成"三种 arm 的均值", 语义就错了)
    bucket = {}
    for r in rows:
        if r["aug"] != "basic" or r["distill"] != "0":
            continue
        v = fnum(r.get(metric))
        if v is None:
            continue
        bucket.setdefault((r["arch"], r["mode"], int(r["data"])), []).append(v)
    # statistics.stdev 需要 >=2 个样本; 单 seed 时 std=None -> 图上不画误差棒 (W7 补种子后自动出现)
    return {k: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else None, len(v))
            for k, v in bucket.items()}

# ==== 4. 落盘工具 ====
def save_both(fig, stem):
    """
    同时存 png 与 pdf (计划 D4: F1/F3 要 .png 与 .pdf 两版)

    参数:
        fig (Figure): matplotlib 图对象
        stem (str): 不含扩展名的路径
    返回:
        list[str]: 写出的文件路径
    """
    out = []
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        # bbox_inches="tight" -> 裁掉多余白边, 避免 300dpi 大图四周空一圈
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        out.append(path)
    plt.close(fig)                      # 循环里反复画图必须关, 否则内存与句柄累积
    return out

# ==== 5. C1 阶梯图 ====
def draw_ladder(ax, rows, metric, ylabel):
    """
    在一个坐标轴上画某指标的全部"架构-策略"折线

    参数:
        ax (Axes); rows (list[dict]); metric (str); ylabel (str)
    返回:
        int: 画出的折线条数 (0 = 该指标无数据)
    """
    agg = agg_baseline(rows, metric)
    for arch, mode in sorted({(k[0], k[1]) for k in agg}):
        xs = [t for t in TIERS if (arch, mode, t) in agg]
        if not xs:
            continue
        ys = [agg[(arch, mode, t)][0] for t in xs]
        es = [agg[(arch, mode, t)][1] or 0.0 for t in xs]
        # errorbar(...) -> 折线 + 误差棒; yerr=0 时不画棒 (单 seed 的正常形态)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    color=MODE_COLORS.get(mode, "gray"),
                    linestyle="--" if arch in DASH_ARCHS else "-",
                    label=f"{arch}-{mode}")
    ax.set_xlabel("train subset (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(TIERS))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    return len({(k[0], k[1]) for k in agg})


def make_f1(rows):
    """
    C1: 先按计划出"3 个 subplot 并排"的一张, 再按 W4 约定每人一张 (含 gap)

    参数:
        rows (list[dict])
    返回:
        list[str]: 写出的文件路径
    """
    paths = []
    label_of = dict(LADDER_METRICS)

    # —— 5.1 计划 D4 的规格: 三个有效读数并排 ——
    fig, axes = plt.subplots(1, len(LADDER3), figsize=(16, 4.6))
    for ax, metric in zip(axes, LADDER3):
        draw_ladder(ax, rows, metric, label_of[metric])
        ax.set_title(metric, fontsize=10)
    fig.suptitle(f"{TAG} C1 ladder (dashed = vits1k start ablation)", fontsize=11)
    fig.tight_layout()
    paths += save_both(fig, os.path.join(OUT_DIR, "F1_ladder"))

    # —— 5.2 W4 约定: 每个指标单独一张 (独立指标共画布会让读者误以为它们可比) ——
    for i, (metric, label) in enumerate(LADDER_METRICS, start=1):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        draw_ladder(ax, rows, metric, label)
        ax.set_title(f"{TAG} C1 ladder - {metric}", fontsize=10)
        fig.tight_layout()
        paths += save_both(fig, os.path.join(OUT_DIR, f"F1_ladder_{i}_{metric}"))
    return paths

# ==== 6. C3 消融图 ====
def _delta(rows, arch, mode, tier, a_key, b_key):
    """
    取两个 arm 的 top1_val 之差 (a - b), 用于柱顶标注

    参数:
        rows (list[dict]); arch/mode (str); tier (int)
        a_key/b_key (str): ARM_KEYS 里的短名
    返回:
        float | None
    """
    key_of = {k[0]: (k[1], k[2]) for k in ARM_KEYS}
    ra = select_arm(rows, arch, mode, tier, *key_of[a_key])
    rb = select_arm(rows, arch, mode, tier, *key_of[b_key])
    if ra is None or rb is None:
        return None
    va, vb = fnum(ra["top1_val"]), fnum(rb["top1_val"])
    return None if va is None or vb is None else va - vb


def make_f3(rows, arch="deitt", mode="Z2"):
    """
    C3: 维度 D 的 2x2 消融 (每档一根基线 / 一根强增强 / 一根蒸馏 / 一根双开)

    参数:
        rows (list[dict])
        arch (str): 固定 deitt (计划: 只翻开关, 不换结构)
        mode (str): 固定 Z2 (DeiT 配方本身是全参训练)
    返回:
        list[str]: 写出的文件路径
    """
    fig, axes = plt.subplots(1, len(ABL_TIERS), figsize=(13, 5.2))
    for ax, tier in zip(axes, ABL_TIERS):
        vals, labels, colors, tops = [], [], [], []
        for short, aug, dis in ARM_KEYS:
            r = select_arm(rows, arch, mode, tier, aug, dis)
            v = fnum(r["top1_val"]) if r is not None else None
            vals.append(v if v is not None else 0.0)
            labels.append(short)
            colors.append(ARM_COLOR[short])
            be = int(r["best_epoch"]) if r is not None else -1
            tops.append(be >= TOP_EPS)
        xs = range(len(vals))
        # 贴顶组换斜纹: 它的 Δ 只能当下界读 (口径 29), 必须在图上肉眼可见
        bars = ax.bar(xs, vals, color=colors, width=0.62,
                      hatch=["//" if t else "" for t in tops], edgecolor="black", linewidth=0.6)
        for x, b, v, short, t in zip(xs, bars, vals, labels, tops):
            ax.text(x, v + 0.6, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
            ax.text(x, v * 0.5, ARM_LABEL[short].replace(" + ", "\n+ ").replace(" (", "\n("),
                    ha="center", va="center", fontsize=7, color="white")
            if t:
                ax.text(x, v + 2.6, "topped\n(delta = lower bound)", ha="center",
                        fontsize=6.5, color="crimson")
        base = vals[0]
        d_aug = _delta(rows, arch, mode, tier, "strong", "baseline")
        d_aug_t = _delta(rows, arch, mode, tier, "both", "distill")
        d_dis = _delta(rows, arch, mode, tier, "distill", "baseline")
        d_dis_s = _delta(rows, arch, mode, tier, "both", "strong")
        ymax = max(vals) * 1.22
        ax.set_ylim(0, ymax)
        note = (f"Delta aug |no teacher = {d_aug:+.2f}\n"
                f"Delta aug |teacher    = {d_aug_t:+.2f}\n"
                f"Delta distill |basic   = {d_dis:+.2f}\n"
                f"Delta distill |strong  = {d_dis_s:+.2f}")
        ax.text(0.99, 0.985, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))
        ax.axhline(base, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels)
        ax.set_ylabel("val top-1 (%)")
        ax.set_title(f"{arch}-{mode}-{tier}  (baseline dotted line)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{TAG} C3 ablation - flip one switch at a time", fontsize=11)
    fig.tight_layout()
    return save_both(fig, os.path.join(OUT_DIR, "F3_ablation"))

# ==== 7. F2 训练曲线 (补充) ====
def make_f2(rows, arch="deitt", mode="Z2"):
    """
    F2: 6 个新臂 + 2 个基线 的训练曲线 (loss 与 val top-1 配对, 两档各一对)

    参数:
        rows (list[dict]); arch/mode (str)
    返回:
        list[str]: 写出的文件路径
    """
    # 为什么 loss 与 top-1 必须同图: 只看 loss 降不判过拟合, 只看 top-1 不判欠训练
    #   (W4 报告 §6 的同一逻辑); 这里两列配对正是为了让"强增强收敛更慢"可见
    fig, axes = plt.subplots(len(ABL_TIERS), 2, figsize=(13, 7))
    for row_i, tier in enumerate(ABL_TIERS):
        for short, aug, dis in ARM_KEYS:
            eid = f"{arch}-{mode.lower()}-{tier}-s42" + (f"-{aug}" if aug != "basic" else "") \
                  + ("-distill" if dis == "1" else "")
            h = read_history(eid)
            if h is None:
                continue
            axes[row_i][0].plot(h["epoch"], h["train_loss"], color=ARM_COLOR[short], label=short)
            axes[row_i][1].plot(h["epoch"], [v * 100.0 for v in h["val_top1"]],
                                color=ARM_COLOR[short], label=short)
            r = select_arm(rows, arch, mode, tier, aug, dis)
            if r is not None and int(r["best_epoch"]) >= 0:
                be = int(r["best_epoch"])
                axes[row_i][1].plot([be], [fnum(r["top1_val"])], marker="*", markersize=13,
                                    color=ARM_COLOR[short], markeredgecolor="black")
        axes[row_i][0].set_xlabel("epoch"); axes[row_i][0].set_ylabel("train loss")
        axes[row_i][0].set_title(f"{tier}% - train loss", fontsize=10)
        axes[row_i][1].set_xlabel("epoch"); axes[row_i][1].set_ylabel("val top-1 (%)")
        axes[row_i][1].set_title(f"{tier}% - val top-1  (star = best_epoch, rolled back)", fontsize=10)
        for ax in axes[row_i]:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{TAG} train curves - {arch}-{mode} (8 arms)", fontsize=11)
    fig.tight_layout()
    return save_both(fig, os.path.join(OUT_DIR, "F2_curves"))

# ==== 8. F4 副读数 (口径 24) ====
def make_f4(rows, arch="deitt", mode="Z2"):
    """
    F4: 蒸馏两臂的 主口径(cls) 与 副读数(cls+dist_avg) 对比

    参数:
        rows (list[dict]); arch/mode (str)
    返回:
        list[str]: 写出的文件路径
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = range(len(ABL_TIERS))
    width = 0.2
    for i, (short, aug, dis) in enumerate([("distill", "basic", "1"), ("both", "strong", "1")]):
        main, sub = [], []
        for tier in ABL_TIERS:
            r = select_arm(rows, arch, mode, tier, aug, dis)
            main.append(fnum(r["top1_val"]) if r is not None else 0.0)
            d = fnum(r.get("top1_val_distavg")) if r is not None else None
            sub.append(d if d is not None else 0.0)
        ax.bar([x + (i * 2 - 1.5) * width for x in xs], main, width=width,
               color=ARM_COLOR[short], label=f"{short}: main (cls)")
        ax.bar([x + (i * 2 - 0.5) * width for x in xs], sub, width=width,
               color=ARM_COLOR[short], alpha=0.5, hatch="xx",
               label=f"{short}: secondary (cls+dist_avg)")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{t}%" for t in ABL_TIERS])
    ax.set_ylabel("val top-1 (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"{TAG} distill arms - main vs secondary eval token (primary uses cls)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7.5)
    fig.tight_layout()
    return save_both(fig, os.path.join(OUT_DIR, "F4_distill_token"))

# ==== 9. 自检 (图 vs 表的一致性) ====
def check_all(rows):
    """
    对账: 抽检值必须与 CSV 一致 + 阶梯图折点不缺

    参数:
        rows (list[dict])
    返回:
        bool: 全部通过
    """
    bad = []
    for key, want in GOLDEN.items():
        if len(key) == 3:
            r = select_arm(rows, key[0], key[1], key[2], "basic", "0")
            got = fnum(r["top1_val"]) if r is not None else None
        else:
            r = select_arm(rows, key[0], key[1], key[2], key[3], key[4])
            got = fnum(r["top1_val"]) if r is not None else None
        if got != want:
            bad.append(f"{key}: want {want} got {got}")
    # 每条基线折线必须正好 3 个点 (少一个点就是"静默掉组", 图上只会少一小段, 肉眼难发现)
    agg = agg_baseline(rows, "top1_val")
    for arch, mode in sorted({(k[0], k[1]) for k in agg}):
        n = sum(1 for t in TIERS if (arch, mode, t) in agg)
        if n != len(TIERS):
            bad.append(f"{arch}-{mode}: 只有 {n}/{len(TIERS)} 个档位")
    if bad:
        for b in bad:
            print(f"[check] MISMATCH {b}")
    return not bad

# ==== 10. 主流程 ====
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_rows()
    print(f"[fig] rows = {len(rows)} (expect 42 = W4 36 组 + W5-D3 6 组)")

    paths = []
    paths += make_f1(rows)          # C1
    paths += make_f3(rows)          # C3
    paths += make_f2(rows)          # 补充: 曲线
    paths += make_f4(rows)          # 补充: 副读数

    for p in paths:
        print(f"[fig] {p}  ({os.path.getsize(p)} B)")
    ok = check_all(rows)
    print(f"[fig] figures = {len(paths)} files (expect 16 = 8 张 x png/pdf)")
    print("[check]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
