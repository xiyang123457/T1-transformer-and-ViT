<#
L1 — 批跑启动器 (W4 用, 支持两批)

用途:
    顺序调用 `run_matrix.py` 跑完一批实验, 全程输出重定向到 logs/ 下的单个日志文件,
    支持后台运行 + 随时 tail 查看。本身不做训练/落盘逻辑, 只做编排 (与 run_matrix.py 职责不重叠)。

    -Phase main  : 主矩阵 18 组 = {r50,vits,deitt} × {Z0,Z1,Z2} × {25,100}, seed 42   [已完成 2026-10-06]
    -Phase extra : W4 两条"允许的加跑" 13 组 =                                           [已完成 2026-10-06]
                   ① 50% 档 9 组 = {r50,vits,deitt} × {Z0,Z1,Z2} × {50}
                   ② vits1k 起点消融 4 组 = vits1k × {Z1,Z2} × {25,100}
    -Phase w5d1  : W5-D1 补 vits1k 起点消融剩余 5 组 =                                  [已完成 2026-10-07]
                   vits1k × {Z0} × {25,50,100} + vits1k × {Z1,Z2} × {50}
                   (W5 计划 §一 更正: Z0 是"起点影响最大"的一档 —— NCM 直接建在预训练特征上,
                    补它才能回答 "vits 的 98.33% 是 Transformer 架构, 还是 21k 预训练")
    -Phase w5d3  : W5-D3 维度 D 消融 6 组 (W5 计划 §六 A 组) =
                   deitt × Z2 × {25,100} × {strong, distill, strong+distill}
                   (固定 arch=deitt / mode=Z2, 只翻「增强」与「teacher」两个开关 ——
                    同结构单开关才叫消融, 见计划 §二 科学目标)
                   ⚠ 6 组必须落成 6 个不同 eid (靠 -strong / -distill 后缀); 否则第 2 组起被 [skip],
                     你以为跑了 6 组实际只有 1 组 (W5 坑位 10)

输入:
    参数 -Phase ("main" | "extra" | "w5d1" | "w5d3"), 默认 main。组清单写死在下面的「矩阵定义」分节, 便于改档位后重跑。

⚠ 本文件含中文, **必须存为 UTF-8 带 BOM**:
    Windows PowerShell 5.1 会按系统 ANSI(代码页 936) 解析**无 BOM** 的 UTF-8。此时若某行中文字符串
    之前的 ASCII 字节数是**奇数**, 收尾引号会被当成多字节字符的第二个字节吞掉 -> 字符串未闭合 ->
    整个脚本解析失败。2026-10-07 实际踩到 (3 个语法错误, 批跑连日志都不生成)。改完本文件记得补 BOM。
输出:
    - `logs/matrix_<时间戳>.log`   本次批跑的完整终端输出 (每组一段)
    - `results/{eid}.json`         每组明细 (由 run_matrix.py 落盘)
    - `results/{eid}_history.csv`  逐 epoch 曲线 (Z0 档无, 属正常)
    - `experiments.csv`            26 列汇总表, 追加本批行数 (表头由第一组自动创建)

跑完应看到的自检数字:
    - main (2026-10-06 实测): `done=18 failed=0`, 耗时 95.96 min, `experiments.csv` 19 行
    - extra (预期):            `done=13 failed=0`, 耗时 ~63 min, `experiments.csv` 32 行
      extra 逐组预估 (依据主矩阵实测: 耗时主要由 epoch 数决定, 而非模型大小):
        Z0 × 3 档 50%        秒级 (r50 0.25 / vits 0.6 / deitt 0.5 min)
        vits1k Z1/Z2 × {100,25}   4 组 ≈ 18 min     (计划书给的量级)
        50% 档 Z1/Z2 × 3 架构     6 组 ≈ 45 min     (r50 各 ~5.8 / vits 各 ~6.0 / deitt 各 ~9.5)

怎么验证跑对了:
    1) `(Get-ChildItem results\*.json | Where-Object Name -notlike "*-dry*").Count` -> 31 (18 + 13)
    2) `(Import-Csv experiments.csv).Count` -> 31, 每行 26 列
    3) 日志末尾应打印 `[batch] done=13 skipped=0 failed=0`
    4) `vits1k` 那 4 组的 `preproc` 列应为 `vit_small/augreg_in1k/0.5-0.5` (与 vits 的 21k+1k 区分)
    5) 中断后续跑: 重跑本脚本, 已完成的组打印 `[skip]` 而不重新训练 (断点续跑)
#>

param(
    # 变量 Phase: str, 选哪一批
    #   示例值: "w5d1"
    #   为什么用参数而非多个脚本: 各批共用同一套前置检查/断点续跑/日志逻辑, 避免复制粘贴导致行为漂移
    [ValidateSet("main", "extra", "w5d1", "w5d3")]
    [string]$Phase = "main"
)

$ErrorActionPreference = "Continue"

# ==== 1. 常量与路径 ====
# 变量 PY: str, 训练用的解释器绝对路径
#   示例值: "D:/anaconda/envs/pytorch/python.exe"
#   为什么写死绝对路径: 批跑在后台无人值守, 不能依赖 conda activate 的 shell 状态
$PY = "D:/anaconda/envs/pytorch/python.exe"

# 变量 ROOT: str, 项目根
#   示例值: "d:/learning project/T1 transformer and ViT"
#   为什么用绝对路径: 脚本可能被从任意目录调用, 相对路径会让 results/ 落到别处
$ROOT = "d:/learning project/T1 transformer and ViT"

# 变量 SEED: int, 全局随机种子
#   示例值: 42
#   为什么: 全周固定同一个种子, 组间差异才能归因于架构/策略/数据量而非随机性 (口径 15)
$SEED = 42

$LOG_DIR = Join-Path $ROOT "logs"
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "matrix_$Phase`_$STAMP.log"

# ==== 2. 矩阵定义 ====
# 设计决策: 用 "arch,mode,tier" 字符串列表显式声明每一组, 而不是嵌套 for 循环
#   为什么: ① 组顺序可以按"便宜的先跑"手工排 (早期就能暴露问题);
#           ② 清单能与计划书的「逐组清单」逐行对上, 便于审计;
#           ③ extra 批的 13 组无法用规则的笛卡尔积表达 (50% 档 9 组 + vits1k 4 组)
$MAIN_LIST = @(
    "r50,Z0,25",   "r50,Z0,100",   "r50,Z1,25",   "r50,Z1,100",   "r50,Z2,25",   "r50,Z2,100",
    "vits,Z0,25",  "vits,Z0,100",  "vits,Z1,25",  "vits,Z1,100",  "vits,Z2,25",  "vits,Z2,100",
    "deitt,Z0,25", "deitt,Z0,100", "deitt,Z1,25", "deitt,Z1,100", "deitt,Z2,25", "deitt,Z2,100"
)

$EXTRA_LIST = @(
    # —— ① 50% 档 9 组 (先跑 3 个 Z0, 秒级即可确认 50% 子集能正常加载) ——
    "r50,Z0,50", "vits,Z0,50", "deitt,Z0,50",
    # —— ② vits1k 起点消融 4 组 (口径 4: 把"预训练数据量"这一混杂量化出来) ——
    "vits1k,Z1,100", "vits1k,Z2,100", "vits1k,Z1,25", "vits1k,Z2,25",
    # —— ③ 50% 档的 6 个训练组 (最贵的放最后, 前面出错能早停) ——
    "r50,Z1,50", "r50,Z2,50",
    "vits,Z1,50", "vits,Z2,50",
    "deitt,Z1,50", "deitt,Z2,50"
)

# —— W5-D1 批 (2026-10-07): 补 vits1k 起点消融剩余 5 组 ——
#   为什么 Z0 也要补: W4 计划原写「配 Z1/Z2, 因为 Z0 是全零训练无关起点」——**这句是错的**。
#     Z0 的 NCM 直接建在**预训练特征**上, 起点对它的影响**最大**而非最小。
#   顺序: 3 个 Z0 (秒级, 且是本周最关键的数字) -> Z1-50 -> Z2-50 (训练组)
$W5D1_LIST = @(
    "vits1k,Z0,25", "vits1k,Z0,50", "vits1k,Z0,100",
    "vits1k,Z1,50", "vits1k,Z2,50"
)

# —— W5-D3 批 (2026-10-07): 维度 D 消融 6 组 (计划 §六 A 组) ——
#   格式: "arch,mode,tier,aug,distill" —— 后两段就是维度 D 的两个开关 (口径 25 增强 / 口径 26 蒸馏)
#   顺序: 先 25% 档 (每格最便宜, 早暴露问题), 同档内 strong -> distill -> 双开; 再 100% 档
#   为什么显式列 6 行而不写笛卡尔积: 只翻这两个开关, 不扩到 Z1 / 不扩到别的架构 (计划 §六)
$W5D3_LIST = @(
    "deitt,Z2,25,strong,0",
    "deitt,Z2,25,basic,1",
    "deitt,Z2,25,strong,1",
    "deitt,Z2,100,strong,0",
    "deitt,Z2,100,basic,1",
    "deitt,Z2,100,strong,1"
)

# 变量 LIST: array, 本批要跑的组清单 (由 -Phase 选)
#   示例值: @("vits1k,Z0,25", ...)
#   为什么用 switch: 三批以上时 if/else 嵌套难读且易漏分支
$LIST = switch ($Phase) {
    "main"  { $MAIN_LIST }
    "extra" { $EXTRA_LIST }
    "w5d1"  { $W5D1_LIST }
    "w5d3"  { $W5D3_LIST }
}
$N = $LIST.Count

# 变量 GROUPS: 对象数组, 解析后的 (arch, mode, tier, aug, distill) 五元组
#   示例值: @{Arch="deitt"; Mode="Z2"; Tier=25; Aug="strong"; Distill="1"}
#   为什么先解析: 循环里要分别取用各字段, 每次切分字符串既慢又容易错
#   向后兼容: 老清单 (main/extra/w5d1) 只写 3 段 -> 后两段取默认 basic / 0, 行为与改造前逐位一致
$GROUPS = @()
foreach ($item in $LIST) {
    $p = $item.Split(",")
    $aug = if ($p.Count -ge 4) { $p[3] } else { "basic" }
    $dis = if ($p.Count -ge 5) { $p[4] } else { "0" }
    $GROUPS += [pscustomobject]@{ Arch = $p[0]; Mode = $p[1]; Tier = [int]$p[2]; Aug = $aug; Distill = $dis }
}

# ==== 3. 前置检查 ====
# 设计决策: 开跑前把"能提前发现的问题"全查掉, 而不是让它跑 20 分钟后在第 7 组才炸
#   Test-Path: 判断路径是否存在, 返回 $True/$False
if (-not (Test-Path $PY)) { Write-Host "[fatal] 找不到解释器: $PY"; exit 1 }
foreach ($f in @("config.py", "data.py", "protocol.py", "run_matrix.py")) {
    if (-not (Test-Path (Join-Path $ROOT $f))) { Write-Host "[fatal] 缺文件: $f"; exit 1 }
}
foreach ($t in ($GROUPS | Select-Object -ExpandProperty Tier -Unique)) {
    $sub = Join-Path $ROOT "subsets\tier_$t.json"
    if (-not (Test-Path $sub)) { Write-Host "[fatal] 缺子集索引: $sub (先跑 make_subsets.py)"; exit 1 }
}

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
Set-Location $ROOT

# 变量 HF_HUB_OFFLINE: str, "1" = 强制离线
#   示例值: "1"
#   为什么: 每组都要新建 timm 模型, 联网查权重会偶发卡住 (W2-D4 踩过, 下载停在 0 字节)
$env:HF_HUB_OFFLINE = "1"
# 变量 PYTHONUNBUFFERED: str, "1" = 关掉 stdout 缓冲
#   为什么: 后台跑时若带缓冲, 日志会攒一大块才落盘, tail 看不到实时进度
$env:PYTHONUNBUFFERED = "1"

"[batch] start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | phase=$Phase | log=$LOG" | Tee-Object -FilePath $LOG
"[batch] groups = $N (phase=$Phase)" | Tee-Object -FilePath $LOG -Append

# —— 预训练权重缓存探针 (2026-10-07 新增; 本体在 cache_probe.py) ——
# 设计决策: 探针逻辑抽成独立 .py, 不在本脚本里内嵌 here-string。为什么:
#   ① Windows PowerShell 5.1 的 here-string 要求 CRLF 行尾, 本仓库文件是 LF ->
#      内嵌 here-string 会让**整个脚本解析失败**(2026-10-07 实际踩到: 3 个语法错误, 批跑直接不启动);
#   ② 独立文件可单测 (`python cache_probe.py vits1k`), 也便于 W5/W6 复用。
# 背景: W4 曾因 HF_HUB_OFFLINE=1 + vits1k 的 augreg_in1k 权重未缓存, 让 4 组「秒失败」,
#   而且是跑完翻日志才发现 —— 开跑前探一次就能挡住。
$probeArchs = @($GROUPS | Select-Object -ExpandProperty Arch -Unique)
& $PY "cache_probe.py" $probeArchs *>> $LOG
if ($LASTEXITCODE -ne 0) {
    "[fatal] 有架构的预训练权重不在本地缓存 (见上); 先下载再跑, 否则该架构的组会秒失败" | Tee-Object -FilePath $LOG -Append
    exit 1
}
"[probe] 本批全部架构的权重均可离线加载" | Tee-Object -FilePath $LOG -Append

# —— 蒸馏前置自检 (2026-10-07 新增) ——
# 设计决策: 只在"本批含蒸馏组"时才跑; 其余批次零开销
# 为什么必须挡在训练之前: distill.py 的自检里有两条**只能真跑一次才验得出来**的东西 ——
#   ① 手写蒸馏损失 / 两路 logits 与 timm 官方的数值对齐 (验收 3);
#   ② 坑位 1 的分级断言: strong 臂的 teacher top1 必须 < 0.92 (实测 83-85%)。
#      这是"teacher 在线前向到底接上了没"的唯一照妖镜 —— 若不接, 蒸馏退化成第二次 CE,
#      Δ蒸馏 结构性恒为 0, 而 6 组会"看起来很正常"地白跑一小时 (口径 27)
if (($GROUPS | Where-Object { $_.Distill -eq "1" }).Count -gt 0) {
    "[probe] 本批含蒸馏组 -> 先跑 distill.py 自检" | Tee-Object -FilePath $LOG -Append
    & $PY "distill.py" *>> $LOG
    if ($LASTEXITCODE -ne 0) {
        "[fatal] distill.py 自检未通过 (见上); 蒸馏组会白跑或退化成第二次 CE, 先修再跑" | Tee-Object -FilePath $LOG -Append
        exit 1
    }
    "[probe] distill.py 自检通过 (数值对齐 + 在线前向分级断言)" | Tee-Object -FilePath $LOG -Append
}

# ==== 4. 逐组执行 (带断点续跑) ====
# 设计决策: 断点续跑在本脚本判断 (Test-Path results/{eid}.json), 而不是只靠 run_matrix 的 --no-resume 开关
#   为什么: 本脚本要能在整批中断后直接重跑; 已完成组的判定标准统一为"json 已存在"
$done = 0
$skipped = 0
$failed = 0
$idx = 0
$tBatch = Get-Date

foreach ($g in $GROUPS) {
    $idx++
    # 变量 eid: str, 实验编号 (必须与 config.ProtocolCfg.eid 规则完全一致: 全小写 + 口径后缀)
    #   示例值: "vits1k-z2-100-s42" / "deitt-z2-100-s42-strong-distill"
    #   为什么脚本里自己拼: 要在调用 run_matrix 之前就知道目标文件名, 才能做断点续跑判断
    #   为什么必须复刻后缀: 后缀漏一个 -> 6 个 D 组落进同一个 eid -> 第 2 组起 [skip] (坑位 10)
    $eid = "$($g.Arch)-$($g.Mode.ToLower())-$($g.Tier)-s$SEED"
    if ($g.Aug -ne "basic") { $eid += "-$($g.Aug)" }
    if ($g.Distill -eq "1") { $eid += "-distill" }
    $json = Join-Path $ROOT "results\$eid.json"

    if (Test-Path $json) {
        $skipped++
        "[$idx/$N] [skip] $eid (已存在 $eid.json, 断点续跑跳过)" | Tee-Object -FilePath $LOG -Append
        continue
    }

    "[$idx/$N] [run ] $eid  @ $(Get-Date -Format 'HH:mm:ss')" | Tee-Object -FilePath $LOG -Append
    $t0 = Get-Date
    # 调用: & $PY run_matrix.py --arch <a> --mode <m> --data <t> --seed 42
    #   返回值: 无 (退出码在 $LASTEXITCODE, 0=成功)
    #   作用  : 跑完一组协议并在 results/ 与 experiments.csv 落盘
    #   关键参数: --arch/--mode/--data/--seed 与 config.ProtocolCfg 字段一一对应
    #   坑    : 输出里既有 stdout 也有 stderr (tqdm 走 stderr),
    #           所以用 *>> 把「所有流」追加进日志; 只写 > 会丢掉进度条
    # 变量 callArgs: array, 传给 run_matrix.py 的完整参数
    #   为什么先拼数组再用 @callArgs 展开: 只有"要不要带 aug/distill"随组变化;
    #     直接字符串拼接容易给 --distill 这种无值开关多带一个参数
    $callArgs = @("run_matrix.py", "--arch", $g.Arch, "--mode", $g.Mode,
                  "--data", $g.Tier, "--seed", $SEED)
    if ($g.Aug -ne "basic") { $callArgs += @("--aug", $g.Aug) }
    if ($g.Distill -eq "1") { $callArgs += "--distill" }
    & $PY @callArgs *>> $LOG
    # 变量 LASTEXITCODE: int, 上一个原生进程的退出码
    #   示例值: 0 (成功) / 1 (异常)
    #   为什么必须查: PowerShell 不会因原生进程失败而中断, 不查就会"静默跑完全部但一半没落盘"
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)

    if ($code -eq 0 -and (Test-Path $json)) {
        $done++
        "[$idx/$N] [ok  ] $eid  ({0} min)" -f $mins | Tee-Object -FilePath $LOG -Append
    } else {
        $failed++
        "[$idx/$N] [FAIL] $eid  exit=$code  ({0} min) -- 详见上方 traceback" -f $mins | Tee-Object -FilePath $LOG -Append
    }
}

# ==== 5. 收尾汇总 ====
$csvRows = 0
if (Test-Path (Join-Path $ROOT "experiments.csv")) {
    # Import-Csv: 把 CSV 读成对象数组; .Count 即数据行数 (不含表头)
    $csvRows = (Import-Csv (Join-Path $ROOT "experiments.csv")).Count
}
$elapsed = [math]::Round(((Get-Date) - $tBatch).TotalMinutes, 2)
"[batch] done=$done skipped=$skipped failed=$failed total=$N | 耗时 $elapsed min" | Tee-Object -FilePath $LOG -Append
# 变量 expectRows: hashtable, 各批跑完后 experiments.csv 应有的数据行数
#   示例值: @{main=18; extra=31; w5d1=36; w5d3=42}
#   为什么逐批写死: 批跑无人值守, 收尾必须自己报"应该多少 / 实际多少", 否则漏跑看不出来
$expectRows = @{ main = 18; extra = 31; w5d1 = 36; w5d3 = 42 }
"[batch] experiments.csv 数据行 = $csvRows (phase=$Phase 跑完后应为 $($expectRows[$Phase]))" | Tee-Object -FilePath $LOG -Append
"[batch] end   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $LOG -Append

if ($failed -gt 0) { exit 1 } else { exit 0 }
