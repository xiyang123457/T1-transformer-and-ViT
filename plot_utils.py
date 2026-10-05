"""
W3-D2 辅助 — 训练曲线绘制 (train loss / val top-1 随 epoch)

用途: 把训练历史画成图并存盘, 作为报告素材 + 直观判断是否收敛
输入: history 字典 { "epoch":[...], "train_loss":[...], "val_top1":[...] } (三者等长)
输出: PNG 文件 (默认 notes/teacher_curves.png; 目录不存在会自动创建)

跑完应看到的自检数字:
    - 图上左子图 train loss 整体下降; 右子图 val top-1 上升后趋稳
    - 右曲线终点数值 == 训练日志最后一行的 val_top1 (一致才说明画对了)
"""
# ==== 1. 依赖导入 ====
import os                                # 自动创建输出目录 (避免 FileNotFoundError)
import matplotlib
matplotlib.use("Agg")                    # 无界面后端: 只存文件、不弹窗, 避免阻塞
import matplotlib.pyplot as plt          # 画图

# ==== 2. 绘制函数 ====
def plot_history(history, out_path="notes/teacher_curves.png"):
    """
    画 train loss 与 val top-1 随 epoch 的曲线

    参数:
        history (dict): 键 'epoch' / 'train_loss' / 'val_top1', 值为等长 list
        out_path (str): 保存路径 (所在目录不存在会自动创建)
    返回:
        None (副作用: 写出一张 PNG)
    """
    epochs = history["epoch"]
    # 变量 epochs: list[int], 训练轮号
    #   示例值: [0, 1, 2, ..., 19]
    #   为什么用它当 x 轴: 让两条曲线共享同一时间轴, 便于对照
    losses = history["train_loss"]
    # 变量 losses: list[float], 每轮训练平均损失; 示例值: [4.6, 3.1, ...]; 观察收敛
    val_accs = history["val_top1"]
    # 变量 val_accs: list[float], 每轮验证 top-1; 示例值: [0.12, 0.55, ..., 0.93]

    # plt.subplots(1, 2, figsize=(12,4)) -> (fig, axes)
    #   作用: 建 1 行 2 列两个子图 (左 loss, 右 acc)
    #   关键参数: figsize 图尺寸(英寸); axes 是长度 2 的数组
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左子图: 训练损失
    axes[0].plot(epochs, losses, marker="o", color="tab:red")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("train loss"); axes[0].grid(alpha=0.3)

    # 右子图: 验证 top-1
    axes[1].plot(epochs, val_accs, marker="o", color="tab:blue")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val top-1")
    axes[1].set_title("val top-1"); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    # os.makedirs(dirname, exist_ok=True) -> None
    #   作用: 递归创建输出目录; 已存在不报错
    #   关键参数: dirname 取路径的目录部分; 若为空串(只给文件名)则回退当前目录 "."
    #   坑: 不建目录时 savefig 会抛 FileNotFoundError (上次就是这个)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # fig.savefig(path, dpi=120) -> None
    #   作用: 存图; dpi 越大越清晰 (默认 100)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)                        # 关图释放内存 (循环里反复画时必须)
    print(f"figure saved: {out_path}")

# ==== 怎么验证跑对了 ====
# 1) 运行后 notes/teacher_curves.png 存在 (目录不存在也会被自动建出来);
# 2) 右曲线终点数值 == 训练日志最后一行 val_top1;
# 3) 左曲线整体下降、右曲线上升后趋平 -> 曲线合理
