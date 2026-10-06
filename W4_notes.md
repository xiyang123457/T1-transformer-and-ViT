# W4 notes

> 只记三件事：开跑前要做的、已修的、已知但**决定不修**的局限。

## 一、开跑前两条动作

1. **先 commit**：`git add -A && git commit -m "W4-D1: 补 W3 遗留 + 修两处口径 bug"`
   否则 18 组 json 里的 `git_commit` 会全指向 W3 的 `82f834c`（不含本次实验代码），溯源列等于失效。
2. **重跑 dry 组**：`python run_matrix.py --arch deitt --mode Z2 --data 10 --seed 42 --dry-run --no-resume`
   一次顶两用：端到端验证两处修复 + 覆盖修复前生成的残留 `results/deitt-z2-10-s42-dry.json`。
   ⚠ 必须带 `--no-resume`，否则被断点续跑静默跳过（这个坑也会让 D1"同组重跑数字一致"的自检变成假的）。

## 二、已修（2026-10-06，均已实测确认）

- **Bug A**：`run_protocol` 恒传 `build_transform(spec, True, ...)` → Z0 误用随机增强，类中心混入噪声、与 val 的 center crop 分布不匹配。已改为 `cfg.train_aug`（Z0=False）。实测 Z0·10% 档三架构差 **7–11 点**，远超 3 点门槛。
- **Bug B**：DeiT 蒸馏版 `forward_head(feat, pre_logits=True)` 返回的是 **`(CLS+dist)/2` 的 token 均值**，不是 `norm(CLS)`。已改为显式 `model.norm(feat[:, 0])`。实测新路径与目标 max|diff| = **0.0**，旧路径差 **6.4e-01**（logits 级）。

## 三、已知局限（决定**不修**，仅记录）

| # | 局限 | 影响 | 处理 |
|---|---|---|---|
| 1 | W5 蒸馏源沿用 `checkpoints/teacher_resnet50.pt`（W3 产物，训练时未开确定性开关） | 与主表 `r50-z2-100-s42`（走 protocol 重跑）可能有 <1 点差异 | 两者配置相同、仅确定性开关不同，W5 报告声明即可 |
| 2 | CSV `mode` 列写大写 `Z2`，清单 A1 定义是小写 `z2` | 纯书写差异，不影响读表 / 画图 / 脚本 | 保留 |
| 3 | teacher ckpt 内无 `git_commit` 字段（口径 B3-② 三件套差一件） | 无法追溯 teacher 是哪版代码训的 | W8 写进局限 |
| 4 | 未实现梯度累积（口径 13 的 OOM 降级路径无代码，`degraded` 只能手填） | 实测 peak_mem = 0.675GB（DeiT·Z2·batch32），8GB 下不会 OOM | 视为"已验证不触发"，W8 写进局限 |
| 5 | `teacher_train.py` 已废（调 `get_dataset(..., train_aug=True)`，现接口是 `get_dataset(split, transform)`） | 该脚本跑不起来；teacher 改由 protocol 产出 | 不再使用，W8 删除或归档 |

## 四、两条解读提醒

- **`recall_min5_val` 在 val 上大概率贴 0**（val 每类仅 10 张 → 单类 recall 只有 0/10/…/100 共 11 档，最差 5 类极易全 0）。实现无误，但 W8 跑 test 之前只能当"有没有长尾塌陷"的定性灯，不能当连续指标读。
- **两处修复只做到了函数级验证**（前向路径 + 变换内容），全流程验证由上面第 2 条 dry 重跑承担。
