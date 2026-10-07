"""
migrate_csv.py — 把 experiments.csv 的表头补到当前 COLUMNS (只追加列, 旧行补空)

用途:
    `run_matrix.append_summary` 用 `csv.DictWriter(fieldnames=COLUMNS)` 写行。一旦 COLUMNS 加了新列
    (W5 的 `top1_val_distavg`), 而文件表头仍是旧列数 -> 出现「表头 26 / 数据 27」的**列错位**,
    之后按名取列全部读错 (W5 坑位 4)。本脚本把旧 CSV 一次性重写为当前列定义。
输入:
    experiments.csv (表头列数 <= len(COLUMNS))
输出:
    experiments.csv (表头 = COLUMNS, 旧行缺的列补空串) + 同目录 `.bak` 备份
    (写盘用"先写 .tmp 再 os.replace", 避免中途失败把原表写坏)

跑完应看到的自检数字 (2026-10-07 实测):
    `python migrate_csv.py` ->
        [migrate] 26 -> 27 列; 行数 36 -> 36; 前 26 列逐格一致; 新增列: ['top1_val_distavg']
        [migrate] 备份: experiments.csv.bak

怎么验证跑对了:
    1) 终端打印的行数前后相同 (36 -> 36)、且 "前 26 列逐格一致" (脚本内已逐格 assert);
    2) 看表头: `head -1 experiments.csv` 末列应为 top1_val_distavg;
    3) 迁移后跑一次非 dry 组, 新行列数 == 27 (不再错位)。
"""
# ==== 1. 依赖导入 ====
import argparse                 # 收 --csv, 便于指向别的表
import csv                      # 读写 CSV (newline="" 是 Windows 下防多空行的必需项)
import os                       # 原子替换
import shutil                   # 备份

# ==== 2. 主流程 ====
def main():
    ap = argparse.ArgumentParser(description="experiments.csv 列迁移 (对齐当前 COLUMNS)")
    ap.add_argument("--csv", default="experiments.csv", help="目标汇总表 (默认 experiments.csv)")
    args = ap.parse_args()

    # 变量 COLUMNS: list[str], 唯一列定义来源
    #   为什么从 run_matrix 导入: 列定义必须只有一处 (口径 15), 迁移脚本不得自己复制一份
    from run_matrix import COLUMNS

    with open(args.csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        old_cols = list(reader.fieldnames or [])
        rows = list(reader)

    if old_cols == list(COLUMNS):
        print(f"[migrate] 已是 {len(COLUMNS)} 列, 无需迁移")
        return 0

    # 变量 missing: list[str], 本次要新增的列名 (例 ['top1_val_distavg'])
    #   为什么打印出来: 迁移是"加列"动作, 让人一眼确认加的是预期的那一列
    missing = [c for c in COLUMNS if c not in old_cols]

    # shutil.copyfile -> 覆盖式备份 (同目录 .bak)
    #   作用  : 迁移前留一份原表, 出错可回退
    #   坑    : 连续跑两次会覆盖上一次备份 —— 迁移只在加列时做一次, 可接受
    shutil.copyfile(args.csv, args.csv + ".bak")

    tmp = args.csv + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            # row.get(k, "") : 旧列取原值; 新增列取空串 -> 达到"旧行末列补空"
            writer.writerow({k: row.get(k, "") for k in COLUMNS})

    # —— 写盘前校验: 行数不变 + 旧列逐格一致 (迁移唯一允许的变化是"多了空列") ——
    with open(tmp, "r", encoding="utf-8", newline="") as fh:
        new_rows = list(csv.DictReader(fh))
    assert len(new_rows) == len(rows), f"行数变了: {len(rows)} -> {len(new_rows)}"
    for i, (old_row, new_row) in enumerate(zip(rows, new_rows)):
        for col in old_cols:
            assert (old_row.get(col) or "") == (new_row.get(col) or ""), \
                f"第 {i} 行 '{col}' 列不一致: {old_row.get(col)!r} vs {new_row.get(col)!r}"

    os.replace(tmp, args.csv)     # 原子替换: 成功前原表始终完好
    print(f"[migrate] {len(old_cols)} -> {len(COLUMNS)} 列; 行数 {len(rows)} -> {len(new_rows)}; "
          f"前 {len(old_cols)} 列逐格一致; 新增列: {missing}")
    print(f"[migrate] 备份: {args.csv}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
