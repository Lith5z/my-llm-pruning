"""绘图相关代码（从 pruning.py 拆分，只依赖 matplotlib）。

图片输出文件名的常量也集中在这里。
"""

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，便于直接保存图片
import matplotlib.pyplot as plt

PRUNE_FIG_FILE = "pruning_results.png"
PRUNE_SIZE_FIG_FILE = "pruning_size_results.png"


def plot_results(results, baseline_acc):
    """画准确率 vs 稀疏度。results: [(sparsity, prune_acc, finetune_acc, kept)]"""
    xs = [r[0] for r in results]
    prune_acc = [r[1] for r in results]
    finetune_acc = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(
        baseline_acc, color="gray", linestyle="--", label=f"baseline {baseline_acc:.4f}"
    )
    ax.plot(xs, prune_acc, "o-", color="tab:red", label="After pruning (no fine-tune)")
    ax.plot(xs, finetune_acc, "s-", color="tab:blue", label="After masked fine-tune")
    ax.set_xlabel("Sparsity (fraction pruned per layer)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Deep Compression pruning: accuracy vs sparsity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRUNE_FIG_FILE, dpi=150)
    plt.close(fig)
    print(f"图已保存: {PRUNE_FIG_FILE}")


def plot_size_results(size_results, dense_kb):
    """画模型尺寸/压缩比 vs 稀疏度。size_results: [(sparsity, kept, size_stats)]"""
    xs = [r[0] for r in size_results]
    compression = [r[2]["compression_ratio"] for r in size_results]
    sparse_kb = [r[2]["sparse_kb"] for r in size_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        label=f"Dense baseline ({dense_kb:.0f} KB, 1.0x)",
    )
    ax.plot(
        xs, compression, "o-", color="tab:red", label="Compression ratio (value-only)"
    )
    # 每个点旁标注稀疏模型大小
    for x, c, kb in zip(xs, compression, sparse_kb):
        ax.annotate(
            f"{kb:.0f} KB",
            (x, c),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="tab:red",
        )
    ax.set_xlabel("Sparsity (fraction pruned per layer)")
    ax.set_ylabel("Compression ratio (dense / sparse)")
    ax.set_title("Deep Compression pruning: model size vs sparsity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRUNE_SIZE_FIG_FILE, dpi=150)
    plt.close(fig)
    print(f"图已保存: {PRUNE_SIZE_FIG_FILE}")
