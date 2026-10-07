"""
cache_probe.py — 预训练权重缓存探针 (W5-D1 起, 每个批跑开跑前自动调用)

用途:
    在**强制离线**模式下逐个试建本批要用的 timm 模型。任何架构的权重不在本地 HF 缓存里,
    就立刻以非零退出码中止整批 —— 而不是等它跑起来后一组接一组「秒失败」。
输入:
    架构代号列表 (= config.ARCH_REGISTRY 的键), 例: `python cache_probe.py vits1k vits`
输出:
    stdout 每个架构一行:
        `[probe] CACHE_OK <code>`            权重在本地缓存, 可离线建模型
        `[probe] CACHE_MISS <code> : <异常>`  缺失 -> 需先下载 (退出码 1)
        `[probe] SKIP <code>`               torchvision 架构, 权重随包分发, 无 HF 依赖
    退出码: 全部命中 -> 0; 有任一缺失 -> 1

背景 (为什么要这个探针):
    W4 的 extra 批把 `HF_HUB_OFFLINE=1` 写进了启动器 (防联网卡住), 而 `vits1k` 用的
    `augreg_in1k` 权重当时不在本地缓存 -> 4 个 vits1k 组「秒失败」(LocalEntryNotFoundError),
    是跑完翻日志才发现的。开跑前 30 秒的探针能把这 4 次失败挡在门外。

跑完应看到的自检数字 (2026-10-07 实测):
    `python cache_probe.py vits1k r50 vits deitt` ->
        [probe] CACHE_OK vits1k
        [probe] SKIP r50 (torchvision, 无 HF 依赖)
        [probe] CACHE_OK vits
        [probe] CACHE_OK deitt
    退出码 0

怎么验证跑对了:
    1) `python cache_probe.py vits1k` -> 打印 CACHE_OK、exit 0
    2) 故意传一个不存在的代号 -> `[probe] UNKNOWN_ARCH xxx` 且 exit 1 (探针自身也会拒错)
    3) 断网跑同一个命令 -> 仍然 CACHE_OK (证明语义是"查本地", 不是"顺手下载")
"""
# ==== 1. 依赖导入 ====
import os                            # 设环境变量 (必须在 import timm 之前)
import sys                           # 取命令行参数 + 设退出码

# 变量 HF_HUB_OFFLINE: str, "1" = 强制离线
#   为什么必须在 import timm 之前设: huggingface_hub 在**导入时**读取一次该变量
#   为什么强制离线: 探针的语义是"本地有没有", 绝不能因为缺缓存就触发下载 (那会把 30 秒探针变成慢下载)
os.environ["HF_HUB_OFFLINE"] = "1"

import timm                          # noqa: E402  (必须在设好环境变量之后再导入)

# ==== 2. 探针主体 ====
def check(codes):
    """
    逐个检查架构权重能否从本地缓存加载

    参数:
        codes (list[str]): 架构代号, 例 ["vits1k", "r50"]
    返回:
        tuple[list[str], list[str]]: (缺失的代号, 未知的代号)
    """
    # 延迟导入 config: 保证上面那行 HF_HUB_OFFLINE 先生效, 且探针不依赖 config 的导入副作用
    from config import ARCH_REGISTRY

    missing, unknown = [], []
    for code in codes:
        spec = ARCH_REGISTRY.get(code)
        if spec is None:
            unknown.append(code)
            print(f"[probe] UNKNOWN_ARCH {code}")
            continue
        # library != timm 的架构 (如 r50 走 torchvision) 权重随包分发, 不涉及 HF 下载
        if getattr(spec, "library", "") != "timm":
            print(f"[probe] SKIP {code} ({spec.library}, 无 HF 依赖)")
            continue
        # timm.create_model("model.tag", pretrained=True, num_classes=102)
        #   作用  : 建结构 + 加载该 tag 的预训练权重 (离线时只能吃本地缓存)
        #   关键参数: 模型名用 "型号.tag" 形式, 与 protocol.build_model 完全一致, 保证探的是同一份权重
        #   坑    : 离线且缺缓存 -> LocalEntryNotFoundError / OfflineModeIsEnabled
        try:
            timm.create_model(f"{spec.model_id}.{spec.tag}", pretrained=True, num_classes=102)
            print(f"[probe] CACHE_OK {code}")
        except Exception as exc:                        # noqa: BLE001  任何异常都算"不能离线用"
            missing.append(code)
            print(f"[probe] CACHE_MISS {code}: {type(exc).__name__}")
    return missing, unknown

# ==== 3. 入口 ====
def main(argv):
    """argv: 架构代号列表; 返回进程退出码 (0=全通过)"""
    if not argv:
        print("[probe] 用法: python cache_probe.py <arch_code> [<arch_code> ...]")
        return 2
    missing, unknown = check(argv)
    if missing or unknown:
        print(f"[probe] 结论: 缺缓存 {missing} | 未知代号 {unknown} -> 先处理再开跑")
        return 1
    print("[probe] 结论: 本批全部架构的权重均可离线加载")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
