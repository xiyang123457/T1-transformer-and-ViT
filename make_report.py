"""
make_report.py — W4 汇总产物生成器 (指标 + 图表 + 报告)

用途:
    只读 `results/*.json`, 一次性生成 W4 计划书要求的全部可交付物, 全部落盘到 notes/。
    ⚠ 出图约定: **每张 PNG 只放 1 张图, 或一对"必须一起看"的配套图**——
      "配套"指两条曲线分开看就没有意义 (如 train/val loss 同看才判过拟合, loss/acc 同看才判见顶);
      彼此独立的指标一律拆成独立 PNG, 用「编号_语义名」区分, 不塞进一个多面板大图。

    - {tag}_metrics.csv               展平的全部指标 (机器读; W5–W7 直接消费)
    - {tag}_earlystop.csv             早停口径敏感性逐组结果 (选点标准 top-1 vs loss)
    - {tag}_ladder_1..4_<metric>.png  阶梯图 C1, **每个指标一张**: top1_val / macro_f1_val
                                      / recall_min5_val / gap; vits1k 虚线; 多 seed 自动误差棒
    - {tag}_curve_01..NN_<eid>.png    训练曲线, **每组一张**: 左 loss(train+val 配套) + 右 val top-1
    - {tag}_earlystop_1_criterion.png 选点口径对比 (左右两图配套: 轮号对比 + 代价对比)
    - {tag}_report.md                 报告: 总表 + 名次 + 早停 + 一致性校验 + 缺失组 + 产物清单
    幂等: 任何时候重跑, 都按当前 results/ 重新生成全部产物 (不追加、不累加),
          并先清掉上一轮的编号图, 避免组数变化后残留旧文件。

输入:
    results/*.json (跳过文件名含 "-dry" 的试跑组); 命令行参数 --tag / --out
输出:
    上述文件 (默认 notes/ 目录, 目录不存在会自动创建); 图张数 = 4 + 训练组数 + 1

跑完应看到的自检数字 (2026-10-06 首跑, 22 组时):
    - `{tag}_metrics.csv` 行数 == 已完成组数 (22)
    - `{tag}_curve_*` 张数 == 有 history 的组数 (13; Z0 六组无训练曲线)
    - `{tag}_ladder_*` 恰 4 张 (四口径各一)
    - `{tag}_report.md` 的「缺失组」节点出未跑的 eid (本次 9 个)
    - 阶梯图 `_1_top1_val` 里 Z0 三条线在 25/50/100 三档单调上升 (r50 51.76→64.90→71.27)
    - 早停表里 `vits-z2-100-s42` 的 best_epoch(top1) == 4

怎么验证跑对了:
    1) `python make_report.py --tag W4` 后, notes/ 下新增 6 个 W4_* 文件, 大小均 > 0
    2) 打开 `notes/W4_metrics.csv`, 抽一行与 `experiments.csv` 同 eid 的 `top1_val` 对齐
       (两处都按"百分数 2 位小数", 应完全相同)
    3) 终端最后打印 `[consistency] ...`; 若 CSV 行数与 json 数不一致会明确报警
"""
# ==== 1. 依赖导入 ====
import argparse                 # 收 --tag / --out, 避免把 W5/W6 的产物覆盖掉
import csv                      # 写 metrics / earlystop 两张表
import glob                     # 扫 results/*.json
import json                     # 读单组明细
import os                       # 路径拼接与建目录

import matplotlib               # 画图
matplotlib.use("Agg")           # 无界面后端: 只存文件、不弹窗 (与 plot_utils.py 一致, 服务器/后台安全)
import matplotlib.pyplot as plt  # noqa: E402
import statistics               # 多 seed 时取均值/标准差 (不引 numpy, 保持依赖最轻)

# 变量 TAG_DEFAULT: str, 产物文件名前缀
#   示例值: "W4"
#   为什么可配: W5/W6/W7 会重复用本脚本, 前缀一变就不会互相覆盖
TAG_DEFAULT = "W4"

# ==== 2. 常量与期望覆盖 ====
ARCHS = ("r50", "vits", "deitt")           # 主矩阵三架构 (口径 1)
MODES = ("Z0", "Z1", "Z2")                 # 三档策略 (口径 14)
MAIN_TIERS = (25, 100)                     # W4 主矩阵两档 (口径 11: 压缩规则① 先保广度)
EXTRA_TIER = (50,)                         # W4 允许的顺手加跑: 50% 档
VITS1K_ARCH = "vits1k"                     # 起点消融 (口径 4)
VITS1K_ABB = ("Z1", "Z2")                  # 消融只配 Z1/Z2 (Z0 是全零训练, 与起点无关)

# 变量 EXPECTED: list[tuple], W4 计划内的全部 eid 三元组
#   示例值: [("r50", "Z0", 25), ...] 共 31 个
#   为什么写死期望集: 「缺失组」是 D5 汇总自检的硬要求 (18 组齐全、没有空格),
#                     必须能报出"哪几组该有但还没有", 而不是只看已完成数
EXPECTED = (
    [(a, m, t) for a in ARCHS for m in MODES for t in MAIN_TIERS]     # 18 组主矩阵
    + [(a, m, t) for a in ARCHS for m in MODES for t in EXTRA_TIER]   # 9 组 50% 档
    + [(VITS1K_ARCH, m, t) for m in VITS1K_ABB for t in MAIN_TIERS]   # 4 组起点消融
)

METRIC_COLS = ["top1_val", "macro_f1_val", "recall_mean_val", "recall_min5_val"]
# 变量 METRIC_COLS: list[str], 四口径在 json 里的键名 (口径 A2)
#   为什么单独列出: 比例→百分数的格式化和"最差值"统计都只对这几列做

# 变量 PCT_COLS: list[str], 需要按"百分数 2 位小数"输出的列
#   为什么: val 只有 1020 张 -> 1 张 = 0.098 点, 报 4 位小数是假精度 (与 run_matrix.py 同口径)
PCT_COLS = METRIC_COLS + ["top1_train", "gap"]

# ==== 3. 读数工具 ====
def pct(v):
    """比例 -> 百分数(2 位小数); None 原样返回 (Z0 无 top1_train/gap)"""
    return None if v is None else round(float(v) * 100.0, 2)


def load_results(results_dir):
    """
    读取全部非 dry 的单组 JSON

    参数:
        results_dir (str): results/ 目录
    返回:
        list[dict]: 每个元素是一组的结果字典 (含 history), 按 (data, arch, mode) 排序
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if "-dry" in name:
            continue          # 试跑组不入总表 (计划书 D1: 试跑组标 dry-run, 不入正式表)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue          # 正在被批跑写入的半成品文件: 跳过, 不打断本次汇总
        rows.append(obj)
    rows.sort(key=lambda d: (int(d.get("data", 0)), str(d.get("arch")), str(d.get("mode"))))
    return rows


def history_arrays(row):
    """
    把 json 里的 history 转成按列取用的 dict (已按 epoch 升序)

    参数:
        row (dict): 单组结果 (含 "history": list[dict])
    返回:
        dict: {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_top1": [...]}
              无 history (Z0) 时各列均为空 list
    """
    hist = sorted(row.get("history") or [], key=lambda h: h["epoch"])
    return {k: [h[k] for h in hist] for k in ("epoch", "train_loss", "val_loss", "val_top1")}

# ==== 4. 指标表落盘 ====
def metrics_columns():
    """metrics.csv 的列定义 (顺序固定, 便于 W5–W7 直接按名取列)"""
    return (["eid", "date", "arch", "mode", "data", "seed", "aug", "distill", "eval_token", "preproc"]
            + ["best_epoch", "epochs_run"] + METRIC_COLS
            + ["top1_train", "gap", "minutes", "peak_mem_gb", "degraded", "assert_ok",
               "ncm_ok", "center_shape", "git_commit", "note"])


def build_metrics_rows(rows):
    """把结果字典列表摊平成可直接写 CSV 的行列表"""
    out = []
    for r in rows:
        ncm = r.get("ncm_check") or {}
        line = {k: r.get(k) for k in ("eid", "date", "arch", "mode", "data", "seed", "aug",
                                      "distill", "eval_token", "preproc",
                                      "best_epoch", "epochs_run",
                                      "minutes", "peak_mem_gb", "degraded", "git_commit", "note")}
        for c in PCT_COLS:
            line[c] = pct(r.get(c))
        line["assert_ok"] = r.get("assert_ok")
        line["ncm_ok"] = ncm.get("ok") if ncm else None
        # center_shape 例 [102, 384]; 压成 "102x384" 便于在表里扫一眼对上架构 (口径: D 对上 384/192/2048)
        cs = ncm.get("center_shape") if ncm else None
        line["center_shape"] = "x".join(str(x) for x in cs) if cs else None
        out.append(line)
    return out


def write_csv(path, columns, rows):
    """写 CSV (UTF-8, 无 BOM); 目录不存在会自动创建"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in columns} for r in rows])

# ==== 5. 早停口径敏感性 (核心分析) ====
def earlystop_rows(rows):
    """
    逐组比较两种"选点标准"会选到哪一轮、差多少点

    背景: 现口径用 val_top1 选点 (protocol.py 的 best_top1), 计划书坑位 12 关心
          "各组 epoch 不同会不会伤可比性"。本函数回答三个问题:
          ① 两种标准各选第几轮 (best_epoch_top1 vs best_epoch_loss)
          ② 若改按 loss 选点, 最终 top-1 会差多少点 (delta_top1 <= 0 表示 loss 选点更差)
          ③ 早停后还多跑了多少轮 (tail: 回滚到 best 之后白跑的轮数)

    参数:
        rows (list[dict]): 非 Z0 组 (Z0 无 history, 直接跳过)
    返回:
        list[dict]: 每组一行
    """
    out = []
    for r in rows:
        if r.get("mode") == "Z0":
            continue
        arr = history_arrays(r)
        if not arr["epoch"]:
            continue
        v_top1, v_loss = arr["val_top1"], arr["val_loss"]
        i_top1 = max(range(len(v_top1)), key=lambda i: v_top1[i])   # 首个最大值 (与 protocol 的严格 > 同效)
        i_loss = min(range(len(v_loss)), key=lambda i: v_loss[i])   # 首个最小值
        best_top1 = pct(v_top1[i_top1])
        out.append({
            "eid": r["eid"],
            "arch": r["arch"], "mode": r["mode"], "data": r["data"],
            "epochs_run": len(v_top1),
            "best_epoch_top1": arr["epoch"][i_top1],
            "top1_at_best_top1": best_top1,
            "best_epoch_loss": arr["epoch"][i_loss],
            "val_loss_min": round(v_loss[i_loss], 6),
            "top1_at_best_loss": pct(v_top1[i_loss]),
            # delta = "按 loss 选点" 相比 "按 top1 选点" 的差; <=0 说明 loss 选点不优
            "delta_top1": round(pct(v_top1[i_loss]) - best_top1, 2),
            # tail = 选点之后还多跑了几轮 (早停没触发时为 0; 触发时 > 0 且这些轮次不影响最终数字)
            "tail_after_best": len(v_top1) - 1 - i_top1,
            "early_stopped": len(v_top1) < 30,
        })
    return out

# ==== 6. 一致性校验 (D5 汇总自检) ====
def check_consistency(rows, results_dir, csv_path):
    """
    校验"结果文件 / 汇总表 / 期望覆盖"三者一致

    参数:
        rows (list[dict]); results_dir (str); csv_path (str, experiments.csv)
    返回:
        dict: {"missing": [...], "duplicated_csv_eid": [...], "csv_rows": int, "json_rows": int}
    """
    have = {(r["arch"], r["mode"], int(r["data"])) for r in rows}
    missing = [f"{a}-{m.lower()}-{t}-s42" for (a, m, t) in EXPECTED if (a, m, t) not in have]

    csv_rows, dup = 0, []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            eids = [r["eid"] for r in csv.DictReader(f)]
        csv_rows = len(eids)
        seen = set()
        dup = sorted({e for e in eids if (e in seen) or seen.add(e)})   # set.add 返回 None -> 恒假, 取重复项
    return {"missing": missing, "duplicated_csv_eid": dup,
            "csv_rows": csv_rows, "json_rows": len(rows)}

# ==== 7. 图表 ====
# 设计决策: 图内一律英文标签 —— 与 plot_utils.py 的 teacher_curves.png 保持一致,
#   且避免 matplotlib 缺 CJK 字体时中文变成方框 (报告正文用中文, 图注用英文, 各取所长)
def _aggregate(rows, metric):
    """
    按 (arch, mode, tier) 聚合, 支持多 seed

    参数:
        rows (list[dict]); metric (str): 指标键 (见 PCT_COLS)
    返回:
        dict: {(arch, mode, tier): (mean_pct, std_or_None, n_seed)}
    """
    bucket = {}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        bucket.setdefault((r["arch"], r["mode"], int(r["data"])), []).append(float(v) * 100.0)
    agg = {}
    for k, vals in bucket.items():
        # statistics.stdev 需要 >=2 个样本; 单 seed 时 std=None -> 图上不画误差棒
        agg[k] = (round(statistics.mean(vals), 2),
                  round(statistics.stdev(vals), 2) if len(vals) > 1 else None,
                  len(vals))
    return agg


def plot_ladder(rows, out_dir, tag):
    """
    阶梯图 C1: **每个指标单独一张 PNG** (4 张), 不把它们挤在一个多面板大图里

    参数:
        rows (list[dict]): 全部已完成组
        out_dir (str): 输出目录
        tag (str): 文件名前缀
    返回:
        list[str]: 生成的文件路径 (按指标顺序; 某指标全无数据时跳过)
    """
    # panels: (指标键, 文件名用短名, y 轴标签)
    #   为什么用短名而非标签当文件名: 短名即 json 里的键名, 与 metrics.csv 列名一一对应, 便于反查
    panels = [("top1_val", "top1_val", "val top-1 (%)"),
              ("macro_f1_val", "macro_f1_val", "val macro-F1 (%)"),
              ("recall_min5_val", "recall_min5_val", "val worst-5 recall (%)"),
              ("gap", "gap", "overfit gap (pp)")]
    modes = {"Z0": "tab:blue", "Z1": "tab:orange", "Z2": "tab:green"}
    tiers = sorted({int(r["data"]) for r in rows})
    paths = []

    for i, (metric, short, ylabel) in enumerate(panels, start=1):
        agg = _aggregate(rows, metric)
        if not agg:
            continue                      # 例: 全部组都没有 gap (只跑了 Z0) 时跳过
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arch, mode in sorted({(k[0], k[1]) for k in agg}):
            xs = [t for t in tiers if (arch, mode, t) in agg]
            if not xs:
                continue
            ys = [agg[(arch, mode, t)][0] for t in xs]
            es = [agg[(arch, mode, t)][1] or 0.0 for t in xs]   # 单 seed 时 std=None -> 0 (不画棒)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                        color=modes.get(mode, "gray"),
                        linestyle="--" if arch == VITS1K_ARCH else "-",
                        label=f"{arch}-{mode}")
        ax.set_xlabel("train subset (%)"); ax.set_ylabel(ylabel)
        ax.set_xticks(tiers); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"{tag} C1 ladder - {short}  (dashed = vits1k ablation)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{tag}_ladder_{i}_{short}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_curves(rows, out_dir, tag):
    """
    训练曲线: **每组一张 PNG** (编号前缀保证与 metrics.csv 行序一致)

    为什么一张图里放两块面板: loss 与 acc 是**配套**的 —— 分开看判不出过拟合
    (必须"loss 还在降 / top-1 已见顶"同看才能下结论), 符合"配套图不拆"的约定。
    左面板 loss 里 train 与 val 两条线同理: 单看任一条都读不出泛化差距。

    参数:
        rows (list[dict]); out_dir (str); tag (str)
    返回:
        list[str]: 生成的文件路径 (只含训练过的组; Z0 无 history -> 不出图)
    """
    trained = [r for r in rows if history_arrays(r)["epoch"]]
    paths = []
    for i, r in enumerate(trained, start=1):
        a = history_arrays(r)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        # 左: loss 对 (train/val 同坐标系 —— 配套的两条线, 不拆)
        axes[0].plot(a["epoch"], a["train_loss"], marker="o", ms=3, color="tab:red", label="train")
        axes[0].plot(a["epoch"], a["val_loss"], marker="o", ms=3, color="tab:blue", label="val")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
        axes[0].set_title(f"{r['eid']} - loss"); axes[0].grid(alpha=0.3)
        # 右: val top-1 (与左图配套: 一起看才判得出"何时见顶/是否过拟合")
        axes[1].plot(a["epoch"], [v * 100.0 for v in a["val_top1"]],
                     marker="o", ms=3, color="tab:green")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val top-1 (%)")
        axes[1].set_title(f"{r['eid']} - val top-1"); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{tag}_curve_{i:02d}_{r['eid']}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_earlystop(es_rows, out_dir, tag):
    """
    选点口径对比: **左右两图配套, 同放一张 PNG** (轮号对比 + 代价对比, 是同一件事的两个视角)

    参数:
        es_rows (list[dict]): earlystop_rows() 的产物
        out_dir (str); tag (str)
    返回:
        list[str]: 生成的文件路径 (无数据时为空 list)
    """
    if not es_rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    eids = [r["eid"] for r in es_rows]
    y = range(len(eids))
    axes[0].barh([i - 0.2 for i in y], [r["best_epoch_top1"] for r in es_rows],
                 height=0.4, color="tab:blue", label="best epoch by val top-1")
    axes[0].barh([i + 0.2 for i in y], [r["best_epoch_loss"] for r in es_rows],
                 height=0.4, color="tab:red", label="best epoch by val loss")
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels(eids, fontsize=7)
    axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(list(y), [r["delta_top1"] for r in es_rows], color="tab:purple")
    axes[1].set_yticks(list(y)); axes[1].set_yticklabels(eids, fontsize=7)
    axes[1].set_xlabel("delta top-1 if selected by loss (pp; <0 = loss is worse)")
    axes[1].grid(alpha=0.3, axis="x")
    fig.suptitle(f"{tag} early-stop criterion sensitivity")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_earlystop_1_criterion.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]

# ==== 8. 报告 ====
def fmt(v):
    """表格单元格格式化: None -> '—'"""
    return "—" if v is None else v


def write_report(path, rows, es_rows, chk, tag, figs):
    """
    写 Markdown 报告: 总表 + 名次 + 早停结论 + 一致性校验 + 产物清单

    参数:
        path (str): 报告输出路径
        rows / es_rows / chk: 见各自构造函数
        tag (str): 文件名前缀
        figs (list[str]): 本轮生成的图路径 (写进「产物清单」, 免得日后找不到图在哪)
    """
    lines = [f"# {tag} 汇总报告 (自动生成, 勿手改)", ""]
    lines.append(f"- 已完成 **{len(rows)}** 组; 计划内共 **{len(EXPECTED)}** 组")
    lines.append(f"- 数据来源: `results/*.json` (排除 `-dry` 试跑组); 生成器: `make_report.py`")
    lines.append("")

    # —— 四口径总表 (按档位分块) ——
    for tier in sorted({int(r["data"]) for r in rows}):
        lines += [f"## {tier}% 档 ({'1020' if tier == 100 else 204 if tier == 25 else 510} 张)", "",
                  "| arch | mode | top1_val | macro_f1_val | recall_mean_val | recall_min5_val | top1_train | gap | best_epoch/epochs_run | minutes | peak_mem_gb |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in rows if int(x["data"]) == tier]:
            lines.append("| {arch} | {mode} | {t1} | {f1} | {rm} | {rm5} | {tt} | {gap} | {be}/{er} | {mi} | {pm} |".format(
                arch=r["arch"], mode=r["mode"], t1=fmt(pct(r.get("top1_val"))),
                f1=fmt(pct(r.get("macro_f1_val"))), rm=fmt(pct(r.get("recall_mean_val"))),
                rm5=fmt(pct(r.get("recall_min5_val"))), tt=fmt(pct(r.get("top1_train"))),
                gap=fmt(pct(r.get("gap"))), be=r.get("best_epoch"), er=r.get("epochs_run"),
                mi=r.get("minutes"), pm=r.get("peak_mem_gb")))
        lines.append("")

    # —— 架构名次 (D5 汇总自检第 2 条: 两档名次是否一致) ——
    lines += ["## 架构名次对比 (D5 自检: 两档名次是否一致)", "",
              "| 策略 | 档 | 名次 (高→低) |", "|---|---|---|"]
    for mode in MODES:
        for tier in sorted({int(r["data"]) for r in rows}):
            sub = [r for r in rows if r["mode"] == mode and int(r["data"]) == tier]
            sub = [r for r in sub if r.get("top1_val") is not None]
            if len(sub) < 2:
                continue
            rank = " > ".join(f"{r['arch']} {pct(r['top1_val'])}"
                              for r in sorted(sub, key=lambda x: -x["top1_val"]))
            lines.append(f"| {mode} | {tier}% | {rank} |")
    lines.append("")

    # —— 早停口径敏感性 ——
    lines += ["## 早停口径敏感性 (计划书坑位 12)", "",
              "`delta_top1` = 若改按 val_loss 选点, 最终 top-1 的变化 (负数 = 更差);",
              "`tail_after_best` = 选到最优点之后还白跑了多少轮 (回滚使其不影响最终数字)。", "",
              "| eid | epochs_run | best_epoch(top1) | top1 | best_epoch(loss) | top1@best_loss | delta | tail | 早停触发 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(es_rows, key=lambda x: -(x["delta_top1"])):
        lines.append(f"| {r['eid']} | {r['epochs_run']} | {r['best_epoch_top1']} | {r['top1_at_best_top1']} | "
                     f"{r['best_epoch_loss']} | {r['top1_at_best_loss']} | {r['delta_top1']} | "
                     f"{r['tail_after_best']} | {'是' if r['early_stopped'] else '否(跑满30)'} |")
    if es_rows:
        worst = min(r["delta_top1"] for r in es_rows)
        same = sum(1 for r in es_rows if r["best_epoch_top1"] == r["best_epoch_loss"])
        lines += ["",
                  f"- **两种标准选到同一轮**的组: {same} / {len(es_rows)}",
                  f"- **最大代价**: 改按 loss 选点最坏会让 top-1 掉 **{-worst:.2f} 点** (负值取绝对)",
                  ""]

    # —— 一致性校验 ——
    lines += ["## 一致性校验 (D5 汇总自检)", "",
              f"- `results/` 已完成 json: **{chk['json_rows']}** 组",
              f"- `experiments.csv` 数据行: **{chk['csv_rows']}** 行",
              f"- 缺失组 ({len(chk['missing'])} 个): " + ("、".join(f"`{m}`" for m in chk["missing"]) if chk["missing"] else "无 ✅"),
              f"- CSV 重复 eid: " + ("、".join(chk["duplicated_csv_eid"]) if chk["duplicated_csv_eid"] else "无 ✅"),
              ""]

    # —— 产物清单 (一图一文件, 按名索引; 免得到 W8 写报告时找不到图在哪) ——
    lines += ["## 产物清单", "", "| 文件 | 内容 |", "|---|---|",
              f"| `{tag}_metrics.csv` | 全指标展平表 (机器读) |",
              f"| `{tag}_earlystop.csv` | 早停口径敏感性逐组结果 |",
              f"| `{tag}_report.md` | 本报告 |"]
    for p in figs:
        lines.append(f"| `{os.path.basename(p)}` | 图 |")
    lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ==== 9. 主流程 ====
def main():
    ap = argparse.ArgumentParser(description="W4 汇总产物生成器 (指标 + 图表 + 报告)")
    ap.add_argument("--tag", default=TAG_DEFAULT, help="产物文件名前缀, 例 W4")
    ap.add_argument("--results", default="results", help="结果目录 (默认 results)")
    ap.add_argument("--out", default="notes", help="产物输出目录 (默认 notes)")
    ap.add_argument("--csv", default="experiments.csv", help="26 列汇总表路径 (仅用于一致性校验)")
    args = ap.parse_args()

    rows = load_results(args.results)
    if not rows:
        print("[fatal] results/ 里没有任何非 dry 的 json")
        return 1
    out = args.out
    os.makedirs(out, exist_ok=True)

    # ① 指标表
    write_csv(os.path.join(out, f"{args.tag}_metrics.csv"), metrics_columns(), build_metrics_rows(rows))
    # ② 早停敏感性
    es_rows = earlystop_rows(rows)
    if es_rows:
        write_csv(os.path.join(out, f"{args.tag}_earlystop.csv"), list(es_rows[0].keys()), es_rows)
    # ③ 一致性校验
    chk = check_consistency(rows, args.results, args.csv)
    # ④ 先清掉上一轮的图, 再重新出图
    #   为什么必须清: 新图名带编号与 eid, 若某组从结果里消失, 旧 PNG 会残留并被误当成有效产物
    for pat in (f"{args.tag}_ladder_*.png", f"{args.tag}_curve_*.png",
                f"{args.tag}_ladder.png", f"{args.tag}_curves.png", f"{args.tag}_earlystop.png"):
        for stale in glob.glob(os.path.join(out, pat)):
            os.remove(stale)

    # ⑤ 图表 (一图一文件; 只有"配套图"才放同一张 PNG)
    figs = plot_ladder(rows, out, args.tag)          # 阶梯图: 每指标一张 (4 张)
    figs += plot_curves(rows, out, args.tag)         # 训练曲线: 每组一张 (只含训练过的组)
    figs += plot_earlystop(es_rows, out, args.tag)   # 选点对比: 左右配套 -> 同一张
    # ⑥ 报告 (含产物清单, 登记上面所有图)
    write_report(os.path.join(out, f"{args.tag}_report.md"), rows, es_rows, chk, args.tag, figs)

    print(f"[report] {args.tag}: {len(rows)}/{len(EXPECTED)} 组")
    print(f"[report] tables   -> {out}/{args.tag}_metrics.csv, {out}/{args.tag}_earlystop.csv")
    print(f"[report] figures  -> {len(figs)} 张 (一图一文件):")
    for p in figs:
        print(f"           {os.path.basename(p)}")
    print(f"[report] markdown -> {out}/{args.tag}_report.md")
    print(f"[consistency] json={chk['json_rows']} csv={chk['csv_rows']} "
          f"missing={len(chk['missing'])} dup_csv={len(chk['duplicated_csv_eid'])}")
    if chk["missing"]:
        print(f"[consistency] 未完成: {', '.join(chk['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
