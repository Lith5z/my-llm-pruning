"""绘图相关代码（从 pruning.py 拆分，只依赖 matplotlib）。

图片输出文件名的常量也集中在这里。
"""

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，便于直接保存图片
import matplotlib.pyplot as plt

PRUNE_FIG_FILE = "pruning_results.png"
PRUNE_SIZE_FIG_FILE = "pruning_size_results.png"
SPARSE_SIZE_FIG_FILE = "sparse_size_results.png"
QUANT_SIZE_FIG_FILE = "quant_size_results.png"
QUANT_ACC_FIG_FILE = "quant_accuracy_results.png"
HUFFMAN_SIZE_FIG_FILE = "huffman_size_results.png"
WEIGHT_DIST_FIG_FILE = "weight_distribution.png"

# 三级压缩（论文 Table 2 的对应物）的线型/颜色/图例
STAGE_STYLE = {
    "sparse": ("o--", "tab:red", "Pruning only: CSR/CSC + index diff (fp32 values)"),
    "quant": ("s-", "tab:blue", "+ Weight sharing: 4/5-bit codebook"),
    "huffman": ("^-", "tab:green", "+ Huffman coding"),
}


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


def plot_sparse_size_results(rows, dense_weights_kb):
    """画稀疏存储（值 + 行计数 + index diff）的压缩比 vs 稀疏度。

    rows: [(sparsity, value_kb, full_kb, value_ratio, full_ratio)]
    """
    xs = [r[0] for r in rows]
    value_ratio = [r[3] for r in rows]
    full_ratio = [r[4] for r in rows]
    full_kb = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        xs,
        value_ratio,
        "o--",
        color="tab:red",
        label="Non-zero values only (fp32)",
    )
    ax.plot(
        xs,
        full_ratio,
        "s-",
        color="tab:blue",
        label="+ row counts & index diffs (CSR/CSC)",
    )
    for x, c, kb in zip(xs, full_ratio, full_kb):
        ax.annotate(
            f"{kb:.0f} KB",
            (x, c),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="tab:blue",
        )
    ax.set_xlabel("Sparsity (fraction pruned per layer)")
    ax.set_ylabel("Compression ratio (dense / sparse)")
    ax.set_title(
        f"Deep Compression step 1: sparse storage vs sparsity\n"
        f"(dense conv/fc weights = {dense_weights_kb:.0f} KB)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SPARSE_SIZE_FIG_FILE, dpi=150)
    plt.close(fig)
    print(f"图已保存: {SPARSE_SIZE_FIG_FILE}")


def plot_compression_results(rows, dense_weights_kb, keys, fig_file, title):
    """画各阶段压缩比 vs 稀疏度（dense 权重为分母）。

    rows: [(sparsity, {阶段: KB})]；keys: 要画的阶段（见 STAGE_STYLE），最后一条会标注 KB。
    """
    xs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key in keys:
        style, color, label = STAGE_STYLE[key]
        ratios = [dense_weights_kb / r[1][key] for r in rows]
        ax.plot(xs, ratios, style, color=color, label=label)

    # 只在最后（最强）一条曲线上标注体积，避免图太乱
    style, color, _ = STAGE_STYLE[keys[-1]]
    for x, r in zip(xs, rows):
        ax.annotate(
            f"{r[1][keys[-1]]:.1f} KB",
            (x, dense_weights_kb / r[1][keys[-1]]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("Sparsity (fraction pruned per layer)")
    ax.set_ylabel("Compression ratio (dense weights / compressed)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_file, dpi=150)
    plt.close(fig)
    print(f"图已保存: {fig_file}")


def plot_quant_accuracy(rows, baseline_acc):
    """画权重共享前后的准确率 vs 稀疏度。

    rows: [(sparsity, acc_pruned, acc_quant, acc_quant_finetuned)]
    """
    xs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(
        baseline_acc, color="gray", linestyle="--", label=f"baseline {baseline_acc:.4f}"
    )
    ax.plot(xs, [r[1] for r in rows], "o-", color="tab:red", label="Pruned + fine-tune (fp32)")
    ax.plot(
        xs,
        [r[2] for r in rows],
        "s--",
        color="tab:orange",
        label="+ Weight sharing (before fine-tune)",
    )
    ax.plot(
        xs,
        [r[3] for r in rows],
        "^-",
        color="tab:blue",
        label="+ Centroid fine-tune",
    )
    ax.set_xlabel("Sparsity (fraction pruned per layer)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Deep Compression step 2: weight sharing (4/5-bit codebook)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(QUANT_ACC_FIG_FILE, dpi=150)
    plt.close(fig)
    print(f"图已保存: {QUANT_ACC_FIG_FILE}")


def plot_weight_distribution(weights, centroids, diffs, title_prefix):
    """权重分布 + codebook / index 差值分布（论文 Figure 3/4/5 的对应物）。

    weights/centroids/diffs 都取自同一个代表层。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(weights, bins=100, color="tab:blue", alpha=0.7, label="Non-zero weights")
    for c in centroids:
        ax.axvline(c, color="tab:red", linestyle="-", linewidth=0.8)
    ax.axvline(
        centroids[0],
        color="tab:red",
        linewidth=0.8,
        label=f"k-means centroids (k={len(centroids)})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{title_prefix}: weight distribution and codebook")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    bins = range(0, int(max(diffs.max(), 1)) + 2)
    ax.hist(diffs, bins=bins, color="tab:green", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Index difference (within a row)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{title_prefix}: index diff distribution after pruning")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(WEIGHT_DIST_FIG_FILE, dpi=150)
    plt.close(fig)
    print(f"图已保存: {WEIGHT_DIST_FIG_FILE}")
