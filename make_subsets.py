"""
W4-D1 — 一次性脚本: 落盘四档 train 子集索引 (口径 16⑤ 的不可逆保险)

用途:
    生成 `subsets/tier_{10,25,50,100}.json`——每类抽中的 index 列表 + 抽样种子 + 索引 hash。
    只在"跑实验之前"做一次; 之后 run_matrix 每次跑前会校验 hash, 保证跨组用的是同一批数据。
输入: 无 (读本地 flowers-102)
输出: `subsets/tier_10.json` / `tier_25.json` / `tier_50.json` / `tier_100.json`

跑完应看到的自检数字:
    - 四档大小 = 102 / 204 / 510 / 1020 (每类 1/2/5/10 张)
    - 嵌套 10%⊂25%⊂50%⊂100% 全 True
    - 同 seed 重算 hash 与落盘 hash 全一致 (可复现)
    - 每档每类 min/max 分别为 1/1、2/2、5/5、10/10 (没有空类)
"""
# ==== 1. 依赖导入 ====
import numpy as np                                       # 类别统计
from config import DATA_TIERS, SUBSET_SEED                # 四档百分比 + 子集种子 (唯一来源)
from data import (get_raw_dataset, get_targets, save_tier,   # 数据 + 落盘
                  stratified_subset_indices, index_hash, tier_path)

# ==== 2. 抽样种子 (整周固定; 与 config 的 ProtocolCfg.seed 保持一致) ====
SEED = SUBSET_SEED
# 变量 SEED: int, 子集抽样种子 (取自 config.SUBSET_SEED, 唯一来源)
#   示例值: 42
#   为什么: 子集一旦定下就是"数据集本身"; 换种子等于换数据集 -> 跨组不可比
#   注意: 它与 ProtocolCfg.seed (训练种子) 是两回事, W7 补种子时不改这个

# ==== 3. 主流程 ====
def main():
    ds = get_raw_dataset("train")
    # 变量 ds: torchvision Dataset, train 原始 split (不带 transform, 只为取标签)
    #   示例值: len=1020
    targets = get_targets(ds)
    # 变量 targets: np.ndarray[int64], 每张图的类别 (0..101)
    #   示例值: 长度 1020, 每类恰好 10 张 (E4)
    print(f"[info] train 总数={len(targets)}  类别数={len(np.unique(targets))}")

    records = []
    for tier in DATA_TIERS:
        # data.save_tier(tier, targets, SEED) -> dict
        #   作用: 分层抽样该档 + 落盘 subsets/tier_{tier}.json
        #   关键参数: tier 是百分比 (10/25/50/100); SEED 固定保证可复现
        #   坑: 落盘后不要再改抽样代码 —— 一旦改动, 新子集会与旧落盘不一致 (漂移)
        records.append(save_tier(tier, targets, SEED))

    # ---- 打印四档一览 ----
    print(f"\n{'tier':>5} {'n_total':>8} {'per-class min/max':>18} {'hash(8)':>10}")
    for rec in records:
        counts = [len(v) for v in rec["per_class"].values()]
        print(f"{rec['tier']:>4}% {rec['n_total']:>8d} "
              f"{min(counts):>10d}/{max(counts):<6d} {rec['hash'][:8]:>10}")

    # ---- 校验 1: 嵌套 (10% ⊂ 25% ⊂ 50% ⊂ 100%) ----
    # 注: 打印里用 ASCII "<=" 而不是数学符号 "⊂" —— Windows 控制台默认 GBK, 打不出该字符会崩
    sets = [set(rec["indices"]) for rec in records]
    nested = all(sets[i] <= sets[i + 1] for i in range(len(sets) - 1))
    print(f"\n嵌套校验 10%<=25%<=50%<=100%: {nested}")
    assert nested, "嵌套不成立! 数据量-精度曲线将不可比"

    # ---- 校验 2: 可复现 (用当前代码重算, hash 必须与落盘一致) ----
    for rec in records:
        again = stratified_subset_indices(targets, rec["ratio"], rec["seed"])
        ok = index_hash(again) == rec["hash"]
        print(f"复现校验 tier_{rec['tier']}: {ok}  ({tier_path(rec['tier'])})")
        assert ok, f"tier_{rec['tier']} 重算 hash 与落盘不一致 -> 抽样逻辑漂移"

# ==== 怎么验证跑对了 ====
# 1) 终端四档表: n_total 为 102/204/510/1020, per-class min==max (无空类)
# 2) "嵌套校验 ... True" 且 "复现校验 tier_* : True" 共 4 行
# 3) `dir subsets` 能看到 tier_10.json / tier_25.json / tier_50.json / tier_100.json
# 4) 再跑一次本脚本, 四个 hash(8) 应与上次完全相同
if __name__ == "__main__":
    main()
