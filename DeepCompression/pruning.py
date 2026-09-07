"""
读取 baseline -> 按层剪枝 -> 掩码微调 -> 评测；只处理 conv/linear 的 weight。
画图代码见 plotting.py。
"""

import torch
from model import (
    BATCH_SIZE,
    DATA_PATH,
    MODEL_FILE_NAME,
    NUM_WORKERS,
    MyCNN,
    test,
    test_transforms,
    train_transforms,
)
from plotting import plot_results, plot_size_results
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 实验超参数 ----------------
# 需要扫描的稀疏度：0.5 表示每层剪掉 |w| 最小的 50%
SPARSITIES = [0.5, 0.75, 0.9, 0.95]
# 微调
FINETUNE_EPOCHS = 15
FINETUNE_LR = 0.001
FINETUNE_MOMENTUM = 0.9
FINETUNE_WEIGHT_DECAY = 1e-4

PRUNE_LOG_FILE = "log_prune.txt"
PRUNED_PREFIX = "pruned_s"
MASK_PREFIX = "mask_s"


def prunable_params(model):
    """返回 {参数名: Conv2d/Linear 的 weight}，剪枝与微调都用它对齐 mask。"""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            out[name + ".weight"] = module.weight
    return out


def prune_model(model, sparsity):
    """按层把 |w| <= quantile(|w|, sparsity) 的权重置 0，返回 (masks, kept_ratio)。"""
    params = prunable_params(model)
    masks = {}
    kept_total = all_total = 0
    rows = []
    with torch.no_grad():
        for name, w in params.items():
            total = w.numel()
            if sparsity > 0:
                threshold = torch.quantile(w.abs().flatten(), sparsity)
                mask = w.abs() > threshold
            else:
                threshold = torch.zeros((), device=w.device)
                mask = torch.ones_like(w, dtype=torch.bool)
            kept = int(mask.sum())
            kept_total += kept
            all_total += total
            # 直接把被剪的权重置 0
            w.mul_(mask.to(w.dtype))
            masks[name] = mask
            rows.append((name, kept, total, float(threshold)))

    # 逐层打印
    print(f"--- 剪枝后各层稀疏情况 (sparsity={sparsity}) ---")
    for name, kept, total, thr in rows:
        print(
            f"{name:16s} kept={kept:8d}/{total:<8d} ({kept / total:.4f}) threshold={thr:.5f}"
        )
    kept_ratio = kept_total / all_total
    print(
        f"总保留率: {kept_ratio:.4f}（目标 sparsity={sparsity}，实际剪掉 {1 - kept_ratio:.4f}）"
    )
    return masks, kept_ratio


def fine_tune(
    model,
    masks,
    trainloader,
    testloader,
    epochs=FINETUNE_EPOCHS,
    lr=FINETUNE_LR,
):
    """掩码微调：被剪位置梯度清零且每步后重新置 0，其余参数（含 BN/bias）正常训练。

    返回 history（每轮 train/test 的 loss/acc）。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=FINETUNE_MOMENTUM,
        weight_decay=FINETUNE_WEIGHT_DECAY,
    )
    # [(参数名, weight参数, mask)]
    masked = [(n, p, masks[n]) for n, p in prunable_params(model).items()]

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for data, label in trainloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()

            # 1) 被剪位置的梯度清零
            with torch.no_grad():
                for _, p, mask in masked:
                    p.grad.mul_(mask.to(p.grad.dtype))

            optimizer.step()

            # 2) 被剪位置的权重重新置 0
            with torch.no_grad():
                for _, p, mask in masked:
                    p.data.mul_(mask.to(p.dtype))

            bs = label.size(0)
            total += bs
            running_loss += loss.item() * bs
            _, predicted = output.max(1)
            correct += predicted.eq(label).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # 每个 epoch 在测试集上评测
        test_loss, test_acc = test(model, testloader, criterion)

        # 校验：被剪位置必须仍然全部为 0（mask 完整性）
        with torch.no_grad():
            pruned_nonzero = sum(
                int(((~mask) & (p.data != 0)).sum()) for _, p, mask in masked
            )
        assert pruned_nonzero == 0, (
            f"epoch {epoch}: 被剪位置偏离 0 的权重个数为 {pruned_nonzero}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"FT Epoch:{epoch:3d}/{epochs} | "
            f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
            f"Test Loss:{test_loss:.4f} Acc:{test_acc:.4f}"
        )
    return history


def model_size_stats(model, masks=None):
    """按 fp32 统计 dense / 仅存非零值的 sparse 尺寸与压缩比（不含索引开销）。

    masks=None 表示未剪枝（dense baseline）。
    """
    total_params = sum(p.numel() for p in model.parameters())
    dense_bytes = total_params * 4
    if masks is None:
        nonzero = total_params
    else:
        pruned = prunable_params(model)
        pruned_names = set(pruned)
        nonzero = sum(int(masks[n].sum()) for n in pruned)
        nonzero += sum(
            p.numel() for n, p in model.named_parameters() if n not in pruned_names
        )
    sparse_bytes = nonzero * 4
    return {
        "total_params": total_params,
        "dense_kb": dense_bytes / 1024.0,
        "nonzero_params": nonzero,
        "sparse_kb": sparse_bytes / 1024.0,
        "compression_ratio": dense_bytes / sparse_bytes,
    }


def build_loaders():
    """与 model.py 相同"""
    trainset = datasets.CIFAR10(
        DATA_PATH, train=True, transform=train_transforms, download=False
    )
    testset = datasets.CIFAR10(
        DATA_PATH, train=False, transform=test_transforms, download=False
    )
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return trainloader, testloader


def size_section_text(size_results, base_stats):
    """把尺寸/压缩比结果格式化成 ASCII 文本。"""
    lines = []
    lines.append(
        "model size analysis (fp32, non-zero values only, no index overhead yet):"
    )
    lines.append(
        f"dense baseline: total={base_stats['total_params']} params = "
        f"{base_stats['dense_kb']:.2f} KB"
    )
    header = (
        f"{'sparsity':>8} {'kept':>6} | {'nonzero_params':>14} "
        f"{'sparse_kb':>10} {'compression':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for sparsity, kept, st in size_results:
        lines.append(
            f"{sparsity:>8.2f} {kept:>6.4f} | {st['nonzero_params']:>14d} "
            f"{st['sparse_kb']:>10.2f} {st['compression_ratio']:>10.2f}x"
        )
    return "\n".join(lines)


def main():
    print(f"设备：{device}")
    trainloader, testloader = build_loaders()
    criterion = nn.CrossEntropyLoss()

    # baseline 参考（只评测，不参与剪枝）
    baseline = MyCNN().to(device)
    baseline.load_state_dict(torch.load(MODEL_FILE_NAME, weights_only=True))
    bl_loss, bl_acc = test(baseline, testloader, criterion)
    print(f"\nBaseline 测试集  Loss:{bl_loss:.4f}  Acc:{bl_acc:.4f}\n")
    baseline_stats = model_size_stats(baseline, None)  # dense 尺寸基线

    results = []  # (sparsity, prune_acc, finetune_acc, kept_ratio)
    size_results = []  # (sparsity, kept_ratio, size_stats)
    log_lines = []
    for sparsity in SPARSITIES:
        tag = f"{sparsity:.2f}"
        print("=" * 64)
        print(f">>> sparsity = {sparsity}")
        print("=" * 64)

        # 每个比例都从 baseline 重新加载，保证公平起点
        model = MyCNN().to(device)
        model.load_state_dict(torch.load(MODEL_FILE_NAME, weights_only=True))

        # 剪枝
        masks, kept_ratio = prune_model(model, sparsity)

        # 剪枝后立即评测（未微调）
        p_loss, p_acc = test(model, testloader, criterion)
        print(f"剪枝后立即评测  Loss:{p_loss:.4f}  Acc:{p_acc:.4f}")

        # 掩码微调
        history = fine_tune(model, masks, trainloader, testloader)
        f_loss = history["test_loss"][-1]
        f_acc = history["test_acc"][-1]
        print(f"掩码微调后评测  Loss:{f_loss:.4f}  Acc:{f_acc:.4f}")

        # 模型尺寸/压缩比分析（只统计非零值，未含索引开销）
        size_stats = model_size_stats(model, masks)
        print(
            f"模型尺寸: 非零 {size_stats['nonzero_params']} "
            f"({size_stats['sparse_kb']:.1f} KB)，"
            f"压缩比 {size_stats['compression_ratio']:.2f}x "
            f"(dense baseline {baseline_stats['dense_kb']:.1f} KB)"
        )

        # 保存结果（被剪权重恒为 0 的模型 + mask）
        torch.save(model.state_dict(), f"{PRUNED_PREFIX}{tag}.pth")
        torch.save({k: v.cpu() for k, v in masks.items()}, f"{MASK_PREFIX}{tag}.pth")
        print(f"已保存: {PRUNED_PREFIX}{tag}.pth / {MASK_PREFIX}{tag}.pth")

        results.append((sparsity, p_acc, f_acc, kept_ratio))
        size_results.append((sparsity, kept_ratio, size_stats))
        line = (
            f"sparsity={sparsity:.2f} kept={kept_ratio:.4f} | "
            f"prune_acc={p_acc:.4f} finetune_acc={f_acc:.4f} (baseline {bl_acc:.4f})"
        )
        print("\n" + line)
        log_lines.append(line)

    # 汇总表 + 日志
    print("\n" + "=" * 64)
    print(f"结果汇总（baseline acc = {bl_acc:.4f}）：")
    header = f"{'sparsity':>8} {'kept':>6} | {'prune_acc':>10} {'finetune_acc':>13}"
    print(header)
    print("-" * len(header))
    for sparsity, p_acc, f_acc, kept in results:
        print(f"{sparsity:>8.2f} {kept:>6.4f} | {p_acc:>10.4f} {f_acc:>13.4f}")
    with open(PRUNE_LOG_FILE, "w") as f:
        f.write(f"baseline acc = {bl_acc:.4f}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for sparsity, p_acc, f_acc, kept in results:
            f.write(f"{sparsity:>8.2f} {kept:>6.4f} | {p_acc:>10.4f} {f_acc:>13.4f}\n")
        f.write("\n")
        for line in log_lines:
            f.write(line + "\n")
        # 追加尺寸/压缩比分析
        f.write("\n" + size_section_text(size_results, baseline_stats))
    print(f"\n日志已保存: {PRUNE_LOG_FILE}")

    # 终端打印尺寸/压缩比汇总
    print("\n" + size_section_text(size_results, baseline_stats))

    # 画图
    plot_results(results, bl_acc)
    plot_size_results(size_results, baseline_stats["dense_kb"])


if __name__ == "__main__":
    main()
