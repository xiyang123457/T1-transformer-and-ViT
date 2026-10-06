"""
W4-D1 — 一次性脚本: 生成 env.txt 环境快照 (清单 B3③)

用途:
    把「跑实验时的环境」固化下来 (pip freeze + torch/timm 版本 + GPU/驱动 + nvidia-smi),
    供报告"可复现性"章节引用; 也是排查"换了环境数字变了"的第一手证据。
输入: 无
输出: `env.txt` (UTF-8)

跑完应看到的自检数字:
    - env.txt 存在, 大小 > 1KB
    - 文件内含 `torch:`、`timm:`、`torchvision:`、`cuda:`、`gpu:` 五个键
    - 含 `nvidia-smi` 段 (能看到 `RTX 5060` 与驱动版本, 例 `Driver Version: 5xx.xx`)
    - 含 `pip freeze` 段 (行数 > 30)
"""
# ==== 1. 依赖导入 ====
import platform                     # 操作系统信息
import subprocess                   # 调 nvidia-smi / pip freeze
import sys                          # 解释器路径与版本
from datetime import datetime       # 打时间戳

# ==== 2. 输出路径 ====
OUT_PATH = r"d:/learning project/T1 transformer and ViT/env.txt"
# 变量 OUT_PATH: str, 环境快照落盘路径 (绝对路径, 原始字符串)
#   示例值: "d:/learning project/T1 transformer and ViT/env.txt"
#   为什么: 报告要引用它; 放项目根便于 git 跟踪

# ==== 3. 主流程 ====
def main():
    lines = []
    # datetime.now().isoformat() -> "2026-10-06T15:04:05"
    #   作用: 记录"这台环境是什么时候测的"
    lines.append(f"date: {datetime.now().isoformat()}")
    lines.append(f"python: {sys.version.split()[0]}  ({sys.executable})")
    lines.append(f"platform: {platform.platform()}")

    # torch 段 (torch/torchvision 若缺失则如实记录, 不让脚本崩)
    try:
        import torch
        # torch.__version__ -> "2.9.0+cu130"
        #   作用: 版本 + 编译的 CUDA 版本 (cu130 = CUDA 13.0), 判断 wheel 是否匹配显卡
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda: {torch.version.cuda}")
        lines.append(f"cudnn: {torch.backends.cudnn.version()}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            # get_device_name(0) -> "NVIDIA GeForce RTX 5060"
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
            lines.append(f"gpu_count: {torch.cuda.device_count()}")
            props = torch.cuda.get_device_properties(0)
            lines.append(f"gpu_mem_gb: {round(props.total_memory / 2**30, 2)}")
            lines.append(f"cc: {props.major}.{props.minor}")     # 计算能力 (Blackwell = 12.0)
    except Exception as exc:                                      # noqa: BLE001
        lines.append(f"torch: ERROR {exc}")

    for mod in ("torchvision", "timm"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception as exc:                                  # noqa: BLE001
            lines.append(f"{mod}: ERROR {exc}")

    # ---- nvidia-smi 原始输出 (驱动版本 / 显存 / 当前占用) ----
    # subprocess.run([...], capture_output=True, text=True, timeout=20) -> CompletedProcess
    #   作用: 抓 nvidia-smi 全文, 存原始事实
    #   关键参数: text=True 拿字符串; timeout 防止驱动卡死时脚本挂住
    lines.append("--- nvidia-smi ---")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        lines.append(smi.stdout.strip() or f"(empty, rc={smi.returncode})")
    except Exception as exc:                                      # noqa: BLE001
        lines.append(f"(nvidia-smi 调用失败: {exc})")

    # ---- pip freeze 全文 (完整依赖快照) ----
    lines.append("--- pip freeze ---")
    try:
        frz = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=180)
        lines.append(frz.stdout.strip() or f"(empty, rc={frz.returncode})")
    except Exception as exc:                                      # noqa: BLE001
        lines.append(f"(pip freeze 失败: {exc})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved: {OUT_PATH}")
    # 打印关键几行, 便于当场核对
    for ln in lines:
        if ln.startswith(("torch:", "cuda:", "gpu:", "torchvision:", "timm:")):
            print("  " + ln)

# ==== 怎么验证跑对了 ====
# 1) 终端打印 saved 路径 + torch/cuda/gpu/torchvision/timm 五行;
# 2) `Get-Item env.txt` 大小 > 1KB;
# 3) 打开 env.txt 能看到 "gpu: NVIDIA GeForce RTX 5060" 与 pip freeze 段;
# 4) 换了环境后重跑, date 与版本行会变 -> 说明它确实在记录"当时的"环境。
if __name__ == "__main__":
    main()
