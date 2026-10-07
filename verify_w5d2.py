"""
verify_w5d2.py — D2 改动验证 (静态 + GPU 两级)

用途:
    把 W5-D2 的六处改动的验证收敛成一条命令, 输出可直接贴进交付报告的对照表。
    - `--stage static` (秒级)  : 编译 / eid 后缀 / 脏值拦截 / strong 变换构成 / CSV 迁移 / 导入冒烟
    - `--stage gpu`    (~20min): 三个"必须证明"的运行级结论
        G1 非蒸馏分支无回归 : basic + 无蒸馏 的 dry 组, 四口径必须与 W4 的参考快照**逐位相同**
                              (证明"改 protocol 没改坏 distill=False 的老路径")
        G2 强增强可复现     : strong 臂 --dry-run --no-resume 连跑两次, 四口径必须完全一致
                              (强增强引入新随机源, 口径 29 要求先证明它可复现)
        G3 蒸馏端到端       : --distill 跑一次, json 必须含 top1_val_distavg 且口径字段齐全
输入:
    results/deitt-z2-10-s42-dry.json (G1 的 W4 参考快照; 首次运行会自动另存为 _w4_ref_deitt-z2-10.json)
输出:
    stdout 对照表 + 退出码 (0 = 全过, 1 = 有失败项)
    副作用: 在 results/ 下生成 4 个 dry json (deitt-z2-10 的 basic/strong×2/distill)

跑完应看到的自检数字 (2026-10-07 预期):
    [S1] py_compile 8 个文件        -> OK
    [S2] eid 5 例                  -> deitt-z2-100-s42 / -strong / -distill / -strong-distill / vits-z2-25-s42
    [S3] distill_alpha=1.5         -> 被 ValueError 拦截
    [S4] strong 变换              -> 含 RandAugment: True | 含 RandomErasing: False
    [S5] experiments.csv          -> 26 -> 27 列, 36 行不变
    [G1] 四口径 max|diff| vs W4    -> 0.0
    [G2] 两次强增强 dry 的四口径    -> 完全一致
    [G3] top1_val_distavg          -> 非空 (0~1)

怎么验证跑对了:
    1) `python verify_w5d2.py --stage static` 末行 `[static] PASS`, 退出码 0
    2) `python verify_w5d2.py --stage gpu`    末行 `[gpu] PASS`
    3) G1 必须精确为 0 —— 只要不是 0, 说明 distill=False 分支被改动了, 不许进 D3
"""
# ==== 1. 依赖导入 ====
import argparse                 # 选 stage
import json                     # 读 dry json
import os                       # 路径
import subprocess               # 起子进程跑 run_matrix / migrate_csv
import sys                      # 退出码

# Windows 控制台/管道默认用 GBK 编码; 一旦要打印的字符超出 GBK 就抛 UnicodeEncodeError
#   (实测踩到: 子进程输出里带了一个 GBK 编不了的字符 -> 脚本直接崩在 print 上, 检查全白跑)
#   修法: 把本进程 stdout/stderr 强制 UTF-8 + errors="replace"; 只影响本脚本, 不动全局环境
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 变量 HF_HUB_OFFLINE: str, "1" = 强制离线 (子进程会继承)
#   为什么: 验证脚本要建 timm 模型; 离线可避免每次都发 HEAD 请求 (权重已全部缓存)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 变量 PY: str, 解释器绝对路径 (与 run_matrix_all.ps1 保持一致)
#   为什么写死: 本项目在 conda env "pytorch" 里, 系统 python 没有 torch
PY = r"D:/anaconda/envs/pytorch/python.exe"
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
REF = os.path.join(RESULTS, "_w4_ref_deitt-z2-10.json")

CHANGED = ["config.py", "data.py", "protocol.py", "run_matrix.py", "distill.py",
           "cache_probe.py", "migrate_csv.py", "make_report.py"]

# ==== 2. 工具 ====
def run(cmd, timeout=1800):
    """
    跑一条子进程命令并回传 (退出码, 合并输出)

    参数:
        cmd (list[str]): 命令与参数
        timeout (int): 秒
    返回:
        (int, str)
    """
    # subprocess.run(..., capture_output=True, text=True) -> CompletedProcess
    #   作用  : 拿 stdout+stderr 便于失败时定位
    #   关键参数: encoding="utf-8", errors="replace" —— Windows 控制台默认 GBK, 不指定会解码炸
    #   坑    : 光设 encoding 还不够 —— 子进程自己写 stdout 时也用 GBK, 中文会变成乱码字节。
    #           必须给子进程加 PYTHONIOENCODING/PYTHONUTF8, 否则拿回来的证据表没法读
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def load_json(path):
    """读 json (UTF-8); 文件不存在返回 None"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def key_metrics(d):
    """取用于比对的 8 个字段 (四口径 + 过拟合 + 训练过程)"""
    keys = ["top1_val", "macro_f1_val", "recall_mean_val", "recall_min5_val",
            "top1_train", "gap", "best_epoch", "epochs_run"]
    return {k: d.get(k) for k in keys}


def digest(d):
    """跑 run_matrix 并回传生成的结果 json (读取失败返回 None)"""
    return load_json(os.path.join(RESULTS, d["eid"] + ".json"))

# ==== 3. 静态检查 ====
def stage_static():
    """S1–S5: 不需要 GPU 的检查; 返回 (是否全过, 记录列表)"""
    rows = []

    # —— S1: 编译 (最快能发现语法/缩进问题) ——
    rc, out = run([PY, "-m", "py_compile"] + CHANGED)
    rows.append(("S1 py_compile %d 文件" % len(CHANGED), rc == 0, out.strip().splitlines()[-1] if out.strip() else "OK"))

    # —— S2: eid 后缀 (口径指纹) + W4 既有 eid 不变 ——
    code = ("from config import ProtocolCfg as C;"
            "print(C(arch='deitt',mode='Z2',data=100).eid);"
            "print(C(arch='deitt',mode='Z2',data=100,aug='strong').eid);"
            "print(C(arch='deitt',mode='Z2',data=100,distill=True).eid);"
            "print(C(arch='deitt',mode='Z2',data=100,aug='strong',distill=True).eid);"
            "print(C(arch='vits',mode='Z2',data=25,seed=42).eid)")
    rc, out = run([PY, "-c", code])
    got = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    want = ["deitt-z2-100-s42", "deitt-z2-100-s42-strong", "deitt-z2-100-s42-distill",
            "deitt-z2-100-s42-strong-distill", "vits-z2-25-s42"]
    rows.append(("S2 eid 后缀 + W4 既有 eid", got == want, f"got={got}"))

    # —— S3: distill_alpha 脏值拦截 ——
    code = ("from config import ProtocolCfg as C;"
            "import sys\n"
            "try:\n"
            "    C(distill_alpha=1.5)\n"
            "    print('NOT_BLOCKED')\n"
            "except ValueError as e:\n"
            "    print('BLOCKED', e)")
    rc, out = run([PY, "-c", code])
    rows.append(("S3 distill_alpha 越界拦截", "BLOCKED" in out, out.strip()))

    # —— S4: strong 变换的构成 (含 RandAugment, 不含 RandomErasing) ——
    code = ("import warnings; warnings.filterwarnings('ignore')\n"
            "import timm\n"
            "import data as D\n"
            "from config import ARCH_REGISTRY as R\n"
            "m = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k', pretrained=True, num_classes=102)\n"
            "tb, _ = D.build_transform(R['deitt'], True, model=m, aug='basic')\n"
            "ts, _ = D.build_transform(R['deitt'], True, model=m, aug='strong')\n"
            "print('strong_RandAugment', 'RandAugment' in repr(ts))\n"
            "print('strong_RandomErasing', 'RandomErasing' in repr(ts))\n"
            "print('basic_RandAugment', 'RandAugment' in repr(tb))\n"
            "print('basic_eq_strong', repr(tb) == repr(ts))")
    rc, out = run([PY, "-c", code])
    ok_s4 = ("strong_RandAugment True" in out and "strong_RandomErasing False" in out
             and "basic_RandAugment False" in out and "basic_eq_strong False" in out)
    rows.append(("S4 strong 含 RandAugment / 不含 RandomErasing", ok_s4, out.replace("\n", " | ")))

    # —— S5: CSV 迁移 (26 -> 27 列, 行数不变) ——
    rc, out = run([PY, "migrate_csv.py"])
    rows.append(("S5 experiments.csv 列迁移", rc == 0, out.strip().replace("\n", " | ")))

    # —— S6: timm Mixup 的 target 契约 (回归测试) ——
    #   为什么留这条: 2026-10-07 曾把 one-hot 喂给 Mixup, 让它内部又 one_hot 一次 ->
    #   target 变 (B*C, C), 训练时 RuntimeError (32*102 vs 32)。这条静态检查就是那次失败的
    #   回归用例: 传类号必须得 (B, C); 传 one-hot 必须报错。日后换 timm 版本也能当场发现契约变化
    #   关键认识 (2026-10-07 实测): 误用 one-hot 在 Mixup **这一层不报错** —— 它只静默返回
    #     错形状 (B*C, C), 错误要到 loss 里才炸 (所以第一次才没被静态检查挡住)。
    #     因此回归断言必须查**形状**, 不能指望它抛异常
    code = ("import torch, torch.nn.functional as F\n"
            "from distill import build_mixup\n"
            "mx = build_mixup(102)\n"
            "x = torch.randn(8, 3, 224, 224)\n"
            "y = torch.randint(0, 102, (8,))\n"
            "xm, tg = mx(x, y)\n"
            "print('CLASSID_OK', tuple(xm.shape), tuple(tg.shape), round(float(tg.sum(1).mean()), 4))\n"
            "_, bad = mx(x, F.one_hot(y, 102).float())\n"
            "print('ONEHOT_SHAPE', tuple(bad.shape))")
    rc, out = run([PY, "-c", code])
    ok_s6 = ("CLASSID_OK (8, 3, 224, 224) (8, 102)" in out
             and "ONEHOT_SHAPE (816, 102)" in out)
    rows.append(("S6 Mixup 要类号不要 one-hot (回归)", ok_s6, out.replace("\n", " | ")))

    passed = all(ok for _, ok, _ in rows)
    for name, ok, info in rows:
        print(f"[static] {'PASS' if ok else 'FAIL'} {name:42s} {info[:150]}")
    print(f"[static] {'PASS' if passed else 'FAIL'}")
    return passed

# ==== 4. GPU 检查 ====
def stage_gpu():
    """G1–G3: 需要 GPU 的运行级验证; 返回 (是否全过, 记录列表)"""
    rows = []
    base = os.path.join(RESULTS, "deitt-z2-10-s42-dry.json")
    common = ["--arch", "deitt", "--mode", "Z2", "--data", "10", "--seed", "42", "--dry-run"]

    # —— 准备 W4 参考快照 (只做一次, 之后不再被覆盖) ——
    if not os.path.exists(REF):
        if os.path.exists(base):
            with open(base, "r", encoding="utf-8") as fh:
                ref = json.load(fh)
            with open(REF, "w", encoding="utf-8") as fh:
                json.dump(ref, fh, ensure_ascii=False, indent=1)
            print(f"[gpu] 已另存 W4 参考快照 -> {os.path.basename(REF)}")
        else:
            print("[gpu] FAIL 缺少 W4 参考快照 results/deitt-z2-10-s42-dry.json")
            return False

    # —— G1: 非蒸馏分支无回归 ——
    rc, out = run([PY, "run_matrix.py"] + common + ["--no-resume"])
    got, ref = load_json(base), load_json(REF)
    if rc != 0 or got is None:
        rows.append(("G1 非蒸馏无回归", False, f"rc={rc}; {out.strip().splitlines()[-1][:120]}"))
    else:
        diffs = {k: (None if got.get(k) == ref.get(k) else f"{ref.get(k)} -> {got.get(k)}")
                 for k in key_metrics(ref)}
        bad = {k: v for k, v in diffs.items() if v}
        rows.append(("G1 非蒸馏无回归 (vs W4 快照)", not bad, f"差异字段={bad or '无 (逐位一致)'}"))

    # —— G2: strong 臂连跑两次, 四口径一致 ——
    strong = common + ["--aug", "strong", "--no-resume"]
    rc1, out1 = run([PY, "run_matrix.py"] + strong)
    first = load_json(os.path.join(RESULTS, "deitt-z2-10-s42-strong-dry.json"))
    rc2, out2 = run([PY, "run_matrix.py"] + strong)
    second = load_json(os.path.join(RESULTS, "deitt-z2-10-s42-strong-dry.json"))
    if first is None or second is None:
        # 为什么必须把子进程输出带出来: 上次 G2 只报了 rc=1/1 没有任何诊断信息,
        #   为了拿 traceback 又手动复现了一轮 (等于多烧一次 GPU)。失败时一律附 tail
        tail = " / ".join((out2 or out1).strip().splitlines()[-4:])
        rows.append(("G2 强增强可复现", False, f"rc={rc1}/{rc2}; tail={tail}"))
    else:
        bad = {k: f"{first.get(k)} vs {second.get(k)}" for k in key_metrics(first)
               if first.get(k) != second.get(k)}
        rows.append(("G2 强增强双跑一致", not bad, f"差异={bad or '无'}"))

    # —— G3: 蒸馏端到端 (字段齐全 + 副读数非空) ——
    rc, out = run([PY, "run_matrix.py"] + common + ["--distill", "--no-resume"])
    dis = load_json(os.path.join(RESULTS, "deitt-z2-10-s42-distill-dry.json"))
    if rc != 0 or dis is None:
        rows.append(("G3 蒸馏端到端", False, f"rc={rc}; {out.strip().splitlines()[-1][:120]}"))
    else:
        need = ["aug", "distill", "top1_val_distavg", "top1_val", "macro_f1_val",
                "recall_min5_val", "gap", "assert_ok"]
        lack = [k for k in need if dis.get(k) is None]
        ok = (not lack) and dis.get("distill") == 1 and 0.0 <= float(dis["top1_val_distavg"]) <= 1.0 \
            and bool(dis.get("assert_ok"))
        rows.append(("G3 蒸馏端到端 (副读数+字段)",
                     ok, f"缺字段={lack or '无'}; distavg={dis.get('top1_val_distavg')}; "
                         f"top1={dis.get('top1_val'):.4f}; assert_ok={dis.get('assert_ok')}"))

    passed = all(ok for _, ok, _ in rows)
    for name, ok, info in rows:
        print(f"[gpu] {'PASS' if ok else 'FAIL'} {name:34s} {info[:180]}")
    print(f"[gpu] {'PASS' if passed else 'FAIL'}")
    return passed

# ==== 5. 入口 ====
def main():
    ap = argparse.ArgumentParser(description="W5-D2 改动验证 (静态 + GPU)")
    ap.add_argument("--stage", default="static", choices=["static", "gpu", "all"])
    args = ap.parse_args()
    ok = True
    if args.stage in ("static", "all"):
        ok = stage_static() and ok
    if args.stage in ("gpu", "all"):
        ok = stage_gpu() and ok
    print(f"[verify_w5d2] stage={args.stage} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
