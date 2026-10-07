"""
W4-D1 — 批跑入口: 一条命令跑起任意一组, 结果自动落盘

用途:
    组装 `ProtocolCfg` -> 调 `protocol.run_protocol` -> 落三样东西:
      ① `results/{eid}.json`         单组完整记录 (身份 + 四口径 + gap + 断言 + config 快照)
      ② `results/{eid}_history.csv`  逐 epoch 过程指标 (Z0 无训练 -> 不生成)
      ③ `experiments.csv`            26 列汇总表追加一行 (--dry-run 时不写)
输入: CLI 参数 --arch --mode --data --seed
输出: 上述三个文件 (另可选 --save-detail 落 perclass.csv / confusion.npy)

用法示例:
    python run_matrix.py --arch vits --mode Z2 --data 25 --seed 42
    python run_matrix.py --arch r50  --mode Z0 --data 100 --seed 42 --note "Z0 基线"
    python run_matrix.py --arch deitt --mode Z2 --data 10 --seed 42 --dry-run

跑完应看到的自检数字 (见文件末尾「怎么验证跑对了」):
    - `results/{eid}.json` 里 26 列所需字段齐全 (身份 9 + 四口径 + gap + 资源 + 卫生)
    - `experiments.csv` 行数 = 已跑组数 + 1 (表头)
    - 同命令重跑 -> `--no-resume` 时四口径完全一致; 不加时被"断点续跑"跳过
"""
# ==== 1. 依赖导入 ====
import argparse                                # CLI 参数
import csv                                     # 写汇总表 / history
import json                                    # 写单组明细
import os                                      # 目录与路径 + 环境变量
# 设计决策: 必须在 import torch/timm 之前设置 —— cuBLAS 的确定性 workspace 要在 CUDA 初始化前生效
#   为什么: 不设的话 cuBLAS 用非确定性 kernel, 同一组重跑结果会漂 (D1 自检会挂)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import subprocess                              # 取 git commit 号
import sys                                     # 退出码
from datetime import date                      # 记录跑实验的日期

import numpy as np                             # --save-detail 的逐类统计

from config import AUG_CHOICES, DATA_TIERS, MODES, ProtocolCfg   # 唯一配置源 + 枚举
from protocol import run_protocol                   # 协议主体 (本脚本只做编排与落盘)

# ==== 2. 路径与列定义 ====
PROJECT_ROOT = r"d:/learning project/T1 transformer and ViT"
# 变量 PROJECT_ROOT: str, 项目根 (绝对路径) —— 所有落盘都基于它, 与当前目录无关
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CSV_PATH = os.path.join(PROJECT_ROOT, "experiments.csv")

COLUMNS = [
    # —— 身份 10 列 (清单 A1) ——
    "eid", "date", "arch", "mode", "data", "aug", "distill", "seed", "eval_token", "preproc",
    # —— 训练过程 (A3 汇总) ——
    "best_epoch", "epochs_run",
    # —— 核心四口径 (A2) ——
    "top1_val", "macro_f1_val", "recall_mean_val", "recall_min5_val",
    # —— 过拟合 (A4) ——
    "top1_train", "gap",
    # —— 资源 (A6) ——
    "minutes", "peak_mem_gb", "degraded",
    # —— 卫生 (A5) ——
    "assert_ok", "ncm_check",
    # —— 溯源 ——
    "git_commit", "json_path",
    # —— 结论 (唯一手填, A7) ——
    "note",
    # —— W5 新增 (口径 24): 蒸馏副读数, 非蒸馏架构留空 ——
    "top1_val_distavg",
]
# 变量 COLUMNS: list[str], 汇总表 27 列 (口径 16④: 只许往后追加, 不许改写既有列)
#   示例值: 见上 (26 + 1 = 27)
#   为什么新列追加在"最右"(note 之后)而不是插在四口径旁边:
#     口径 16④ 要求既有列的位置与含义一个都不许动 —— 追加到最右是唯一"零风险"的位置,
#     代价只是列的语义顺序不那么顺眼 (读表按名取列, 不按位置)
#   坑: 加列后必须先迁移旧 CSV (表头 26 / 数据 27 -> 按名取列全错), 用 migrate_csv.py

PCT_COLS = {"top1_val", "macro_f1_val", "recall_mean_val", "recall_min5_val",
            "top1_train", "gap", "top1_val_distavg"}
# 变量 PCT_COLS: set[str], 需要按"百分数 2 位小数"落表的列 (清单 A2 报数精度)
#   为什么: val 只有 1020 张 -> 1 张 = 0.098 点, 报 4 位小数是假精度

# ==== 3. 小工具 ====
def git_commit() -> str:
    """取当前代码的 git 短 commit (溯源列; 取不到就写 unknown)"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                            # noqa: BLE001
        return "unknown"

def pct(v):
    """比例 -> 百分数(2 位小数); None 原样返回"""
    return None if v is None else round(float(v) * 100.0, 2)

def ncm_cell(ncm) -> str:
    """把 ncm_check 压成一行短字符串 (Z1/Z2 为 None -> 空串)"""
    if not ncm:
        return ""
    return (f"shape={ncm['center_shape']};refit={ncm['refit_acc']:.4f};"
            f"knn1={ncm['knn1_acc']:.4f};gap={ncm['gap_ncm_knn1']:.4f};ok={ncm['ok']}")

def write_json(path, obj):
    """写 JSON (UTF-8, 缩进 1)"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def write_history(path, history):
    """写逐 epoch 历史 CSV (A3); history 为空则不写"""
    if not history:
        return False
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    return True

def save_detail(eid, preds, labels, num_classes):
    """
    落逐类 perclass.csv + 混淆矩阵 confusion.npy (口径 18, 仅主表最终配置组)

    参数:
        eid (str); preds/labels (np.ndarray); num_classes (int)
    """
    conf = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, t in zip(preds, labels):
        conf[int(t), int(p)] += 1
    np.save(os.path.join(RESULTS_DIR, f"{eid}_confusion.npy"), conf)
    rows = []
    for c in range(num_classes):
        tp = int(conf[c, c]); fn = int(conf[c].sum() - tp); fp = int(conf[:, c].sum() - tp)
        support = tp + fn
        recall = tp / support if support else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"class": c, "support": support, "recall": round(recall, 6),
                     "precision": round(precision, 6), "f1": round(f1, 6)})
    with open(os.path.join(RESULTS_DIR, f"{eid}_perclass.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def append_summary(row):
    """向 experiments.csv 追加一行 (不存在则先写表头)"""
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in COLUMNS})

# ==== 4. 主流程 ====
def main():
    ap = argparse.ArgumentParser(description="T1 实验矩阵批跑入口 (W4/W5)")
    ap.add_argument("--arch", required=True, choices=["r50", "vits", "vits1k", "deitt"])
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--data", required=True, type=int, choices=list(DATA_TIERS))
    ap.add_argument("--seed", type=int, default=42)
    # —— W5 维度 D 的两个开关 (口径 24–26) ——
    # 为什么必须补: 不补则 6 个 D 组无法表达 aug/distill -> 落进同一个 eid -> 第 2 组起被静默 [skip]
    #   (W5 坑位 10, 本周最易漏); 补上后 eid 会带 -strong / -distill 后缀
    ap.add_argument("--aug", default="basic", choices=list(AUG_CHOICES),
                    help="增强档: basic (W4 口径) / strong (DeiT 配方, 口径 25)")
    ap.add_argument("--distill", action="store_true",
                    help="启用 hard 蒸馏 (需 teacher 检查点; 口径 26/27)")
    ap.add_argument("--note", default="", help="27 列里唯一手填的结论列")
    ap.add_argument("--dry-run", action="store_true", help="试跑: eid 加 -dry 后缀, 不写汇总表")
    ap.add_argument("--no-resume", action="store_true", help="已存在 json 也强制重跑")
    ap.add_argument("--save-detail", action="store_true", help="额外落 perclass/confusion")
    args = ap.parse_args()

    cfg = ProtocolCfg(arch=args.arch, mode=args.mode, data=args.data, seed=args.seed,
                      note=args.note, save_detail=args.save_detail,
                      aug=args.aug, distill=args.distill)
    eid = cfg.eid + ("-dry" if args.dry_run else "")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{eid}.json")
    hist_path = os.path.join(RESULTS_DIR, f"{eid}_history.csv")

    # 断点续跑: 已有该组 json 就跳过; **但必须先比对口径** ——
    #   坑 (W5 坑位 3): 若 eid 相同而 aug/distill 不同 (例如忘了加 --aug), 直接 [skip] 会让人
    #   以为"这组跑过了", 实际产物是旧口径的。所以改成"口径不一致就报错", 不静默跳过
    if os.path.exists(json_path) and not args.no_resume:
        with open(json_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        same = (old.get("aug") == cfg.aug) and (int(old.get("distill", 0)) == int(cfg.distill))
        if not same:
            print(f"[fatal] {eid} 已存在但口径不同: json(aug={old.get('aug')}, "
                  f"distill={old.get('distill')}) vs 本次(aug={cfg.aug}, distill={int(cfg.distill)})")
            print("        -> 拒绝静默跳过; 用 --no-resume 覆盖, 或改用正确的 eid 后缀")
            return 1
        print(f"[skip] {eid} 已完成且口径一致 ({json_path}); 用 --no-resume 强制重跑")
        return 0

    print(f"[run ] {eid}  (arch={cfg.arch} mode={cfg.mode} data={cfg.data} seed={cfg.seed})")
    result = run_protocol(cfg)

    # —— 补溯源列 + 身份补 date ——
    result["eid"] = eid                              # dry-run 时带 -dry 后缀
    result["date"] = date.today().isoformat()
    result["git_commit"] = git_commit()
    result["json_path"] = os.path.relpath(json_path, PROJECT_ROOT).replace("\\", "/")
    result["note"] = args.note                       # 26 列里唯一手填的结论列 (A7)

    # —— 落盘: 明细 JSON (剔除私有 ndarray 字段) ——
    preds, labels = result.pop("_preds", None), result.pop("_labels", None)
    write_json(json_path, result)
    wrote_hist = write_history(hist_path, result.get("history", []))
    if args.save_detail and preds is not None:
        save_detail(eid, preds, labels, cfg.num_classes)

    # —— 落盘: 汇总表一行 (dry-run 不写) ——
    if not args.dry_run:
        row = dict(result)
        for col in PCT_COLS:                          # 比例 -> 百分数 2 位
            row[col] = pct(row.get(col))
        row["ncm_check"] = ncm_cell(result.get("ncm_check"))
        append_summary(row)

    print(f"[done] {eid} | json={'OK' if os.path.exists(json_path) else 'MISS'} "
          f"| history={'OK' if wrote_hist else 'N/A(Z0)'} "
          f"| summary={'skipped(dry-run)' if args.dry_run else 'appended'}")
    return 0

# ==== 怎么验证跑对了 ====
# 1) `python run_matrix.py --arch deitt --mode Z2 --data 10 --seed 42 --dry-run`
#    -> results/deitt-z2-10-s42-dry.json 出现, 且不含 experiments.csv 新行
# 2) 打开该 json, 应能看到: eid/date/arch/mode/data/aug/distill/seed/eval_token/preproc/
#    best_epoch/epochs_run/top1_val/macro_f1_val/recall_mean_val/recall_min5_val/
#    top1_train/gap/minutes/peak_mem_gb/degraded/assert_ok/ncm_check/git_commit/json_path/note
# 3) 同一条命令加 --no-resume 再跑一次 -> 四口径数字与上次完全一致 (种子固定)
# 4) 不加 --dry-run 跑一次 -> experiments.csv 出现表头 + 1 行, 且 width == 26
if __name__ == "__main__":
    sys.exit(main())
