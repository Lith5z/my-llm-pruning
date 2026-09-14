"""
Deep Compression 后续阶段的驱动脚本（在 pruning.py 的掩码微调结果上继续）：

  step 1  sparse : CSR/CSC + index diff 稀疏存储
  step 2  quant  : 权重共享（k-means 量化 + centroid 微调）
  step 3  huffman: 对 index 流/值流做 Huffman 编码

每个阶段都会：加载 pruned_s*.pth -> 编码 -> 落盘 -> 解码回模型，验证权重逐位一致
且测试集准确率不变，然后统计尺寸/压缩比并写 log_compression.txt。
"""

import argparse
import os
import time

import numpy as np
import torch
from huffman import (
    encode_huffman,
    entropy_bits,
    huffman_size,
    load_compressed,
    restore_layers,
    save_compressed,
    symbol_counts,
)
from model import MODEL_FILE_NAME, MyCNN, test
from plotting import (
    HUFFMAN_SIZE_FIG_FILE,
    QUANT_SIZE_FIG_FILE,
    plot_compression_results,
    plot_quant_accuracy,
    plot_sparse_size_results,
    plot_weight_distribution,
)
from pruning import build_loaders
from sparse_format import (
    FILLER_CODE,
    FILLER_VALUE,
    apply_to_model,
    dense_weights_bits,
    encode_model,
    kb,
    load_layers,
    load_state,
    other_output_size,
    save_layers,
    sum_size,
    to_dense,
)
from weight_sharing import (
    CONV_BITS,
    FC_BITS,
    FINETUNE_LR,
    build_shared_model,
    check_layers_match,
    expand_shared_model,
    finetune_centroids,
    nonzero_of,
    quantize_model,
    sync_codebook,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 实验超参数 ----------------
SPARSITIES = [0.5, 0.75, 0.9, 0.95]
PRUNED_PREFIX = "pruned_s"
MASK_PREFIX = "mask_s"
COMPRESSION_LOG_FILE = "log_compression.txt"
SPARSE_PREFIX = "sparse_s"
QUANT_PREFIX = "quant_s"
COMPRESSED_PREFIX = "compressed_s"
QUANT_FINETUNE_EPOCHS = 10
DENSE_BITS = 32  # 未压缩权重按 fp32 记


def load_pruned(tag):
    """加载掩码微调后的模型（权重里被剪位置恒为 0）。"""
    model = MyCNN().to(device)
    model.load_state_dict(torch.load(f"{PRUNED_PREFIX}{tag}.pth", weights_only=True))
    return model


def baseline_acc(testloader, criterion):
    """baseline 参考（只评测，不参与压缩）。"""
    baseline = MyCNN().to(device)
    baseline.load_state_dict(torch.load(MODEL_FILE_NAME, weights_only=True))
    _, acc = test(baseline, testloader, criterion)
    return acc


def layer_table(layers, sparsity):
    """逐层打印稀疏存储细节。"""
    print(f"--- 稀疏存储逐层明细 (sparsity={sparsity}) ---")
    header = (
        f"{'layer':16s} {'kept/total':>13} {'rows':>5} {'row_len':>7} {'ib':>3} "
        f"{'nnz':>7} {'filler':>6} {'maxdiff':>7} "
        f"{'val_KB':>7} {'idx_KB':>7} {'cnt_KB':>7} {'sum_KB':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, layer in layers.items():
        b = layer.size_breakdown()
        total = layer.n_rows * layer.row_len
        print(
            f"{name:16s} {layer.nnz:6d}/{total:<6d} {layer.n_rows:>5} "
            f"{layer.row_len:>7} {layer.index_bits:>3} {layer.nnz:>7} "
            f"{layer.n_filler:>6} {layer.max_diff:>7} "
            f"{kb(b['value']):>7.2f} {kb(b['index']):>7.2f} "
            f"{kb(b['row_nnz']):>7.2f} {kb(layer.size_bits()):>7.2f}"
        )


def verify_roundtrip(model, path, testloader, criterion, ref_acc):
    """从落盘文件读回并解码，校验权重逐位一致 + 准确率不变。"""
    meta, layers = load_layers(path)
    assert abs(meta["acc"] - ref_acc) < 1e-9, "meta 里的准确率与当前评测不一致"

    restored = MyCNN().to(device)
    restored.load_state_dict(model.state_dict())  # bias/BN 等不压缩的参数原样带过来
    apply_to_model(restored, layers)

    max_abs = 0.0
    for name, param in restored.named_parameters():
        if name.endswith(".weight") and name in layers:
            cur = dict(model.named_parameters())[name]
            max_abs = max(max_abs, float((param.detach() - cur.detach()).abs().max()))
    assert max_abs == 0.0, f"解码权重与剪枝模型不一致，最大偏差 {max_abs}"

    _, acc = test(restored, testloader, criterion)
    assert acc == ref_acc, f"解码后准确率变化：{acc} != {ref_acc}"
    return acc, max_abs


def run_sparse(testloader, criterion, sparsities):
    """step 1：编码 -> 落盘 -> 解码校验 -> 尺寸统计。"""
    dense_bits = dense_weights_bits(MyCNN())
    base = MyCNN()
    bias_bn_bytes, buffer_bytes = other_output_size(base)

    lines = []
    lines.append("=== step 1: 稀疏存储（CSR/CSC + index diff） ===")
    lines.append(
        f"conv/fc 权重 dense = {kb(dense_bits):.2f} KB "
        f"({sum(m.weight.numel() for m in base.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))} params x fp32)"
    )
    lines.append(
        f"不压缩部分：bias/BN 参数 = {bias_bn_bytes / 1024:.2f} KB，BN buffer = {buffer_bytes / 1024:.2f} KB"
    )
    lines.append(
        "conv: CSR，行=输出通道，index diff 8 bits；fc: CSC（转置后行=输入特征），index diff 5 bits"
    )

    rows = []  # 画图用
    stat_rows = []  # 汇总表
    for sparsity in sparsities:
        tag = f"{sparsity:.2f}"
        print("=" * 72)
        print(f">>> sparsity = {sparsity}")
        print("=" * 72)

        model = load_pruned(tag)
        ref_loss, ref_acc = test(model, testloader, criterion)
        print(f"剪枝+微调模型  Loss:{ref_loss:.4f}  Acc:{ref_acc:.4f}")

        layers = encode_model(model)
        layer_table(layers, sparsity)

        total_bits, parts = sum_size(layers)
        nnz = sum(layer.nnz for layer in layers.values())
        total_weights = sum(layer.n_rows * layer.row_len for layer in layers.values())
        kept = nnz / total_weights
        filler = sum(layer.n_filler for layer in layers.values())
        value_only_kb = kb(nnz * DENSE_BITS)
        full_kb = kb(total_bits)
        print(
            f"合计: kept={kept:.4f} nnz={nnz} filler={filler} | 仅值 {value_only_kb:.2f} KB -> "
            f"含行计数+index {full_kb:.2f} KB "
            f"(值 {kb(parts['value']):.2f} / index {kb(parts['index']):.2f} / "
            f"行计数 {kb(parts['row_nnz']):.2f} KB)"
        )

        path = f"{SPARSE_PREFIX}{tag}.pt"
        t0 = time.time()
        save_layers(path, layers, {"sparsity": sparsity, "acc": ref_acc})
        assert os.path.getsize(path) >= total_bits / 8 - 1, "落盘文件不应小于理论位宽"
        print(
            f"已保存 {path}: {os.path.getsize(path) / 1024:.2f} KB "
            f"(理论 {full_kb:.2f} KB, 打包耗时 {time.time() - t0:.2f}s)"
        )

        acc, max_abs = verify_roundtrip(model, path, testloader, criterion, ref_acc)
        print(f"解码校验通过: 权重最大偏差={max_abs:.0f}  解码后 Acc:{acc:.4f}（与剪枝模型一致）")

        ratio = kb(dense_bits) / full_kb
        print(
            f"压缩比: {ratio:.2f}x（dense conv/fc 权重 {kb(dense_bits):.2f} KB / 稀疏 {full_kb:.2f} KB）"
        )
        rows.append((sparsity, value_only_kb, full_kb, kb(dense_bits) / value_only_kb, ratio))
        stat_rows.append(
            {
                "sparsity": sparsity,
                "kept": kept,
                "nnz": nnz,
                "filler": filler,
                "value_kb": kb(parts["value"]),
                "index_kb": kb(parts["index"]),
                "count_kb": kb(parts["row_nnz"]),
                "full_kb": full_kb,
                "file_kb": os.path.getsize(path) / 1024,
                "acc": acc,
                "ratio": ratio,
            }
        )

    # ---------------- 汇总 ----------------
    header = (
        f"{'sparsity':>8} {'kept':>10} | {'acc':>7} | {'value_KB':>8} {'index_KB':>8} "
        f"{'count_KB':>8} {'weights_KB':>10} {'file_KB':>8} {'ratio':>7}"
    )
    print("\n" + "=" * len(header))
    print("稀疏存储汇总（dense conv/fc 权重 = %.2f KB）:" % kb(dense_bits))
    print(header)
    print("-" * len(header))
    for st in stat_rows:
        print(
            f"{st['sparsity']:>8.2f} {st['kept']:>10.4f} | {st['acc']:>7.4f} | "
            f"{st['value_kb']:>8.2f} {st['index_kb']:>8.2f} {st['count_kb']:>8.2f} "
            f"{st['full_kb']:>10.2f} {st['file_kb']:>8.2f} {st['ratio']:>6.2f}x"
        )

    lines.append(header)
    lines.append("-" * len(header))
    for st in stat_rows:
        lines.append(
            f"{st['sparsity']:>8.2f} {st['kept']:>10.4f} | {st['acc']:>7.4f} | "
            f"{st['value_kb']:>8.2f} {st['index_kb']:>8.2f} {st['count_kb']:>8.2f} "
            f"{st['full_kb']:>10.2f} {st['file_kb']:>8.2f} {st['ratio']:>6.2f}x"
        )
    lines.append(
        "说明：weights_KB = 值流(fp32) + index 差值流 + 行计数；file_KB = 实际落盘文件大小；"
        "ratio = dense 权重 / weights_KB。"
    )

    with open(COMPRESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n日志已追加: {COMPRESSION_LOG_FILE}")

    plot_sparse_size_results(rows, kb(dense_bits))
    return stat_rows


def quant_layer_table(layers, stats, sparsity):
    """逐层打印权重共享细节。"""
    print(f"--- 权重共享逐层明细 (sparsity={sparsity}) ---")
    header = (
        f"{'layer':16s} {'bits':>4} {'k':>3} {'used':>4} {'nnz':>7} "
        f"{'rmse':>9} {'max_err':>9} {'cb_KB':>6} {'val_KB':>7} {'idx_KB':>7} "
        f"{'cnt_KB':>7} {'sum_KB':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, layer in layers.items():
        st = stats[name]
        b = layer.size_breakdown()
        print(
            f"{name:16s} {st['bits']:>4} {st['k']:>3} {st['used']:>4} {st['nnz']:>7} "
            f"{st['rmse']:>9.5f} {st['max_err']:>9.5f} {kb(b['codebook']):>6.3f} "
            f"{kb(b['value']):>7.2f} {kb(b['index']):>7.2f} "
            f"{kb(b['row_nnz']):>7.2f} {kb(layer.size_bits()):>7.2f}"
        )


def uncompressed_state(model, layers):
    """把不压缩的参数（bias / BN 参数与 buffer）摘出来，随文件一起落盘，便于自包含重建。"""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in layers
    }


def verify_quant_file(path, layers, testloader, criterion, ref_acc):
    """只靠落盘文件重建模型：校验码流解码的权重逐位一致 + 准确率不变。"""
    meta, loaded = load_layers(path)
    state = load_state(path)
    assert abs(meta["acc"] - ref_acc) < 1e-9, "meta 里的准确率与当前评测不一致"
    assert set(loaded) == set(layers), "落盘层集合与内存不一致"

    max_abs = 0.0
    for name, layer in layers.items():
        ref = to_dense(layer)
        max_abs = max(max_abs, float((to_dense(loaded[name]) - ref).abs().max()))
    assert max_abs == 0.0, f"解码后的量化权重不一致，最大偏差 {max_abs}"

    restored = MyCNN().to(device)
    restored.load_state_dict(state, strict=False)  # conv/fc 权重由压缩表示重建
    apply_to_model(restored, loaded)
    _, acc = test(restored, testloader, criterion)
    assert acc == ref_acc, f"解码后准确率变化：{acc} != {ref_acc}"
    return acc, max_abs


def run_quant(testloader, trainloader, criterion, sparsities, epochs, skip_finetune=False):
    """step 2：逐层 k-means 量化 -> centroid 微调 -> 落盘 -> 解码校验 -> 尺寸统计。"""
    dense_bits = dense_weights_bits(MyCNN())
    base = MyCNN()
    bias_bn_bytes, buffer_bytes = other_output_size(base)

    lines = ["", "=== step 2: 权重共享（k-means 量化 + centroid 微调） ==="]
    lines.append(
        f"conv {CONV_BITS} bits（{1 << CONV_BITS} centroids）/ fc {FC_BITS} bits"
        f"（{1 << FC_BITS} centroids），逐层独立 k-means（线性初始化 + Lloyd 迭代）"
    )
    lines.append(
        f"centroid 微调：{epochs} epochs，lr={FINETUNE_LR}，momentum=0.9，weight_decay=1e-4，"
        f"BN/bias 一起训练；index 与剪枝结构不变"
    )
    lines.append(
        f"dense conv/fc 权重 = {kb(dense_bits):.2f} KB（分母，与 step 1 相同）；"
        f"不压缩部分：bias/BN 参数 = {bias_bn_bytes / 1024:.2f} KB"
    )

    rows = []  # 画压缩比用
    acc_rows = []  # 画准确率用
    stat_rows = []
    dist_sample = None  # 权重分布图

    for idx, sparsity in enumerate(sparsities):
        tag = f"{sparsity:.2f}"
        print("=" * 72)
        print(f">>> sparsity = {sparsity}（权重共享）")
        print("=" * 72)

        model = load_pruned(tag)
        ref_loss, ref_acc = test(model, testloader, criterion)
        print(f"剪枝+微调模型  Loss:{ref_loss:.4f}  Acc:{ref_acc:.4f}")

        # step 1 的稀疏结构原样复用（值流此刻还是 fp32）
        layers = encode_model(model)
        sparse_bits, _ = sum_size(layers)

        qlayers, qstats = quantize_model(model, layers)
        quant_layer_table(qlayers, qstats, sparsity)
        nnz = sum(layer.nnz for layer in qlayers.values())

        # 量化后（未微调）准确率：此时权重就是 codebook[index]
        shared = build_shared_model(model, qlayers)
        max_abs = check_layers_match(shared, qlayers)
        assert max_abs == 0.0, f"共享权重模块与 codebook 不一致，最大偏差 {max_abs}"
        assert nonzero_of(shared) == nnz, "共享权重模块的非零个数与稀疏结构不一致"
        _, q_acc = test(shared, testloader, criterion)
        print(f"量化后（未微调）Acc:{q_acc:.4f}（相对剪枝模型 {q_acc - ref_acc:+.4f}）")

        history = None
        if skip_finetune:
            qft_acc = q_acc
            print("已跳过 centroid 微调")
        else:
            history, qft_acc = finetune_centroids(
                shared, trainloader, testloader, epochs=epochs
            )
        qlayers = sync_codebook(qlayers, shared)  # 把微调后的 centroid 写回

        max_abs = check_layers_match(shared, qlayers)
        assert max_abs == 0.0, "微调后 codebook 与共享模块不一致"
        # 展开成普通 MyCNN：权重 = codebook[index]，bias/BN 用微调后的值
        dense_model = expand_shared_model(shared)
        _, dense_acc = test(dense_model, testloader, criterion)
        assert dense_acc == qft_acc, f"重建模型准确率 {dense_acc} != {qft_acc}"

        # 落盘 + 从文件读回校验
        path = f"{QUANT_PREFIX}{tag}.pt"
        state = uncompressed_state(dense_model, qlayers)
        save_layers(
            path,
            qlayers,
            {
                "sparsity": sparsity,
                "acc": qft_acc,
                "acc_quant": q_acc,
                "acc_pruned": ref_acc,
                "bits": {"conv": CONV_BITS, "fc": FC_BITS},
            },
            state=state,
        )
        acc, max_abs = verify_quant_file(
            path, qlayers, testloader, criterion, qft_acc
        )

        total_bits, parts = sum_size(qlayers)
        full_kb = kb(total_bits)
        ratio = kb(dense_bits) / full_kb
        file_kb = os.path.getsize(path) / 1024
        rmse = np.sqrt(
            sum(st["nnz"] * st["rmse"] ** 2 for st in qstats.values()) / nnz
        )
        print(
            f"合计: nnz={nnz} RMSE={rmse:.5f} | 量化后 {full_kb:.2f} KB "
            f"(值 {kb(parts['value']):.2f} / index {kb(parts['index']):.2f} / "
            f"行计数 {kb(parts['row_nnz']):.2f} / codebook {kb(parts['codebook']):.3f} KB)"
        )
        print(
            f"已保存 {path}: {file_kb:.2f} KB，解码校验通过（权重最大偏差={max_abs:.0f}，"
            f"Acc {acc:.4f}）  压缩比 {ratio:.2f}x（step 1 稀疏存储为 "
            f"{kb(dense_bits) / kb(sparse_bits):.2f}x）"
        )

        rows.append((sparsity, {"sparse": kb(sparse_bits), "quant": full_kb}))
        acc_rows.append((sparsity, ref_acc, q_acc, qft_acc))
        stat_rows.append(
            {
                "sparsity": sparsity,
                "nnz": nnz,
                "rmse": rmse,
                "sparse_kb": kb(sparse_bits),
                "value_kb": kb(parts["value"]),
                "index_kb": kb(parts["index"]),
                "count_kb": kb(parts["row_nnz"]),
                "codebook_kb": kb(parts["codebook"]),
                "full_kb": full_kb,
                "file_kb": file_kb,
                "acc_pruned": ref_acc,
                "acc_quant": q_acc,
                "acc_finetune": qft_acc,
                "best_ft": None if history is None else history["best_test_acc"][-1],
                "ratio": ratio,
                "sparse_ratio": kb(dense_bits) / kb(sparse_bits),
            }
        )

        if dist_sample is None:  # 用首个稀疏度做分布图（非零权重最多）
            key = max(layers, key=lambda k: layers[k].nnz)
            values = layers[key].values
            dist_sample = (
                f"sparsity={tag} {key}",
                values[values != FILLER_VALUE],
                np.asarray(qlayers[key].codebook),
                layers[key].diff,
            )

    # ---------------- 汇总 ----------------
    header = (
        f"{'sparsity':>8} {'nnz':>7} | {'acc_pruned':>10} {'acc_quant':>10} "
        f"{'acc_quant_ft':>12} | {'value_KB':>8} {'index_KB':>8} {'count_KB':>8} "
        f"{'cb_KB':>6} {'weights_KB':>10} {'file_KB':>8} {'ratio':>8}"
    )
    print("\n" + "=" * len(header))
    print("权重共享汇总（dense conv/fc 权重 = %.2f KB）:" % kb(dense_bits))
    print(header)
    print("-" * len(header))
    for st in stat_rows:
        print(
            f"{st['sparsity']:>8.2f} {st['nnz']:>7} | {st['acc_pruned']:>10.4f} "
            f"{st['acc_quant']:>10.4f} {st['acc_finetune']:>12.4f} | "
            f"{st['value_kb']:>8.2f} {st['index_kb']:>8.2f} {st['count_kb']:>8.2f} "
            f"{st['codebook_kb']:>6.3f} {st['full_kb']:>10.2f} {st['file_kb']:>8.2f} "
            f"{st['ratio']:>7.2f}x"
        )

    lines.append(header)
    lines.append("-" * len(header))
    for st in stat_rows:
        lines.append(
            f"{st['sparsity']:>8.2f} {st['nnz']:>7} | {st['acc_pruned']:>10.4f} "
            f"{st['acc_quant']:>10.4f} {st['acc_finetune']:>12.4f} | "
            f"{st['value_kb']:>8.2f} {st['index_kb']:>8.2f} {st['count_kb']:>8.2f} "
            f"{st['codebook_kb']:>6.3f} {st['full_kb']:>10.2f} {st['file_kb']:>8.2f} "
            f"{st['ratio']:>7.2f}x"
        )
    lines.append("")
    lines.append(
        "对比 step 1（值流 fp32）：值流位宽 32 -> conv 4 / fc 5 bits，index 流与行计数不变，"
        "codebook 每层额外 k x 32 bits。"
    )
    for st in stat_rows:
        lines.append(
            f"sparsity={st['sparsity']:.2f}: step 1 {st['sparse_kb']:.2f} KB "
            f"({st['sparse_ratio']:.2f}x) -> +权重共享 {st['full_kb']:.2f} KB "
            f"({st['ratio']:.2f}x)，加权 RMSE={st['rmse']:.5f}"
            + (
                ""
                if st["best_ft"] is None
                else f"，微调最优 Acc={st['best_ft']:.4f}"
            )
        )

    with open(COMPRESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n日志已追加: {COMPRESSION_LOG_FILE}")

    plot_compression_results(
        rows,
        kb(dense_bits),
        ["sparse", "quant"],
        QUANT_SIZE_FIG_FILE,
        "Deep Compression step 2: model size after weight sharing\n"
        f"(dense conv/fc weights = {kb(dense_bits):.0f} KB)",
    )
    plot_quant_accuracy(acc_rows, baseline_acc(testloader, criterion))
    if dist_sample is not None:
        plot_weight_distribution(*dist_sample)
    return stat_rows


def huffman_layer_table(layers, streams, sparsity):
    """逐层打印 Huffman 编码细节。"""
    print(f"--- Huffman 编码逐层明细 (sparsity={sparsity}) ---")
    header = (
        f"{'layer':16s} {'stream':>7} {'n':>7} {'sym':>4} {'H(b)':>6} {'avg b/s':>8} "
        f"{'fixed_KB':>9} {'huff_KB':>8} {'save':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, layer in layers.items():
        for key, label, fixed_bits in (
            ("values", "values", layer.value_bits),
            ("diff", "index", layer.index_bits),
        ):
            stream = streams[name][key]
            counts = symbol_counts(
                np.asarray(layer.values if key == "values" else layer.diff, dtype=np.int64)
                + stream.offset,
                stream.lengths.size,
            )
            fixed_kb = kb(stream.n_entries * fixed_bits)
            huff_kb = kb(stream.size_bits())
            print(
                f"{name:16s} {label:>7} {stream.n_entries:>7} {int((counts > 0).sum()):>4} "
                f"{entropy_bits(counts):>6.2f} {stream.bits_per_symbol:>8.2f} "
                f"{fixed_kb:>9.2f} {huff_kb:>8.2f} "
                f"{1 - huff_kb / fixed_kb if fixed_kb else 0:>5.1%}"
            )


def run_huffman(testloader, criterion, sparsities):
    """step 3：对 step 2 的 cluster index 流与 index 差值流做 Huffman 编码。"""
    dense_bits = dense_weights_bits(MyCNN())
    base = MyCNN()
    bias_bn_bytes, buffer_bytes = other_output_size(base)

    lines = ["", "=== step 3: Huffman 编码（cluster index 流 + index 差值流） ==="]
    lines.append(
        "每层两条流各自统计频率 -> Huffman 树 -> 规范码（只存码长，不存码字），"
        "位流 MSB-first 打包"
    )
    lines.append(
        f"dense conv/fc 权重 = {kb(dense_bits):.2f} KB（分母，与 step 1/2 相同）；"
        f"不压缩部分：bias/BN 参数 = {bias_bn_bytes / 1024:.2f} KB + BN buffer "
        f"{buffer_bytes / 1024:.2f} KB（随文件原样保存）"
    )
    lines.append("行计数（16 bits）与 codebook（k x fp32）不参与 Huffman，与论文一致")

    rows = []  # 画三级压缩比用
    stat_rows = []
    for sparsity in sparsities:
        tag = f"{sparsity:.2f}"
        print("=" * 72)
        print(f">>> sparsity = {sparsity}（Huffman）")
        print("=" * 72)

        quant_path = f"{QUANT_PREFIX}{tag}.pt"
        meta, layers = load_layers(quant_path)  # step 2 的结果（量化 + centroid 微调）
        ref_acc = meta["acc"]
        state = load_state(quant_path)
        assert state is not None, f"{quant_path} 里没有 bias/BN，无法自包含重建"

        streams = encode_huffman(layers)
        huffman_layer_table(layers, streams, sparsity)

        # 内存里先做一次编解码往返：两条流都要逐位还原
        back = restore_layers(layers, streams)
        for name, layer in layers.items():
            assert np.array_equal(back[name].diff, layer.diff), f"{name}: index 流往返不一致"
            assert np.array_equal(back[name].values, layer.values), f"{name}: 值流往返不一致"

        total_bits, parts = huffman_size(layers, streams)
        nnz = sum(layer.nnz for layer in layers.values())
        full_kb = kb(total_bits)
        sparse_kb = kb(sum_size(encode_model(load_pruned(tag)))[0])  # step 1 的对照值
        print(
            f"合计: nnz={nnz} | Huffman 后 {full_kb:.2f} KB "
            f"(值流 {kb(parts['value']):.2f} + 码长表 {kb(parts['value_table']):.3f} / "
            f"index 流 {kb(parts['index']):.2f} + 码长表 {kb(parts['index_table']):.3f} / "
            f"行计数 {kb(parts['row_nnz']):.2f} / codebook {kb(parts['codebook']):.3f} KB)"
        )

        # 落盘（自包含：Huffman 码流 + bias/BN）-> 只用文件重建模型并评测
        path = f"{COMPRESSED_PREFIX}{tag}.pt"
        t0 = time.time()
        save_compressed(
            path,
            layers,
            streams,
            state,
            {
                "sparsity": sparsity,
                "acc": ref_acc,
                "acc_quant": meta.get("acc_quant"),
                "acc_pruned": meta.get("acc_pruned"),
                "bits": meta.get("bits"),
                "payload_bits": total_bits,
            },
        )
        file_kb = os.path.getsize(path) / 1024
        print(
            f"已保存 {path}: {file_kb:.2f} KB（理论 {full_kb:.2f} KB，"
            f"打包耗时 {time.time() - t0:.2f}s）"
        )

        file_meta, file_state, file_layers = load_compressed(path)
        assert abs(file_meta["acc"] - ref_acc) < 1e-9, "meta 里的准确率与当前评测不一致"
        max_abs = 0.0
        for name, layer in layers.items():
            ref = to_dense(layer)
            max_abs = max(max_abs, float((to_dense(file_layers[name]) - ref).abs().max()))
        assert max_abs == 0.0, f"Huffman 解码后的权重不一致，最大偏差 {max_abs}"

        restored = MyCNN().to(device)
        restored.load_state_dict(file_state, strict=False)
        apply_to_model(restored, file_layers)
        _, acc = test(restored, testloader, criterion)
        assert acc == ref_acc, f"解码后准确率变化：{acc} != {ref_acc}"
        print(
            f"解码校验通过: 权重最大偏差={max_abs:.0f}  Acc {acc:.4f}"
            f"（与 step 2 的 centroid 微调结果一致）"
        )

        ratio = kb(dense_bits) / full_kb
        quant_kb = kb(sum_size(layers)[0])
        print(
            f"压缩比: {ratio:.2f}x（dense {kb(dense_bits):.2f} KB / Huffman {full_kb:.2f} KB）"
            f"，相对 step 2（{quant_kb:.2f} KB）再省 {1 - full_kb / quant_kb:.1%}"
        )

        rows.append(
            (
                sparsity,
                {"sparse": sparse_kb, "quant": quant_kb, "huffman": full_kb},
            )
        )
        stat_rows.append(
            {
                "sparsity": sparsity,
                "nnz": nnz,
                "sparse_kb": sparse_kb,
                "quant_kb": quant_kb,
                "value_kb": kb(parts["value"]),
                "value_table_kb": kb(parts["value_table"]),
                "index_kb": kb(parts["index"]),
                "index_table_kb": kb(parts["index_table"]),
                "count_kb": kb(parts["row_nnz"]),
                "codebook_kb": kb(parts["codebook"]),
                "full_kb": full_kb,
                "file_kb": file_kb,
                "acc": acc,
                "ratio": ratio,
                "quant_ratio": kb(dense_bits) / quant_kb,
            }
        )

    # ---------------- 汇总 ----------------
    header = (
        f"{'sparsity':>8} {'nnz':>7} | {'acc':>7} | {'val_KB':>7} {'valtab_KB':>9} "
        f"{'idx_KB':>7} {'idxtab_KB':>9} {'cnt_KB':>7} {'cb_KB':>6} {'weights_KB':>10} "
        f"{'file_KB':>8} {'ratio':>8}"
    )
    print("\n" + "=" * len(header))
    print("Huffman 汇总（dense conv/fc 权重 = %.2f KB）:" % kb(dense_bits))
    print(header)
    print("-" * len(header))
    for st in stat_rows:
        print(
            f"{st['sparsity']:>8.2f} {st['nnz']:>7} | {st['acc']:>7.4f} | "
            f"{st['value_kb']:>7.2f} {st['value_table_kb']:>9.3f} {st['index_kb']:>7.2f} "
            f"{st['index_table_kb']:>9.3f} {st['count_kb']:>7.2f} {st['codebook_kb']:>6.3f} "
            f"{st['full_kb']:>10.2f} {st['file_kb']:>8.2f} {st['ratio']:>7.2f}x"
        )

    lines.append(header)
    lines.append("-" * len(header))
    for st in stat_rows:
        lines.append(
            f"{st['sparsity']:>8.2f} {st['nnz']:>7} | {st['acc']:>7.4f} | "
            f"{st['value_kb']:>7.2f} {st['value_table_kb']:>9.3f} {st['index_kb']:>7.2f} "
            f"{st['index_table_kb']:>9.3f} {st['count_kb']:>7.2f} {st['codebook_kb']:>6.3f} "
            f"{st['full_kb']:>10.2f} {st['file_kb']:>8.2f} {st['ratio']:>7.2f}x"
        )
    lines.append(
        "说明：weights_KB = 值流 + index 流 + 两张码长表 + 行计数 + codebook；"
        "file_KB = 实际落盘文件大小（含 bias/BN）；ratio = dense 权重 / weights_KB。"
    )
    for st in stat_rows:
        lines.append(
            f"sparsity={st['sparsity']:.2f}: 稀疏 {st['sparse_kb']:.2f} KB -> +权重共享 "
            f"{st['quant_kb']:.2f} KB ({st['quant_ratio']:.2f}x) -> +Huffman "
            f"{st['full_kb']:.2f} KB ({st['ratio']:.2f}x)，Huffman 再省 "
            f"{1 - st['full_kb'] / st['quant_kb']:.1%}，Acc={st['acc']:.4f}"
        )

    with open(COMPRESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n日志已追加: {COMPRESSION_LOG_FILE}")

    plot_compression_results(
        rows,
        kb(dense_bits),
        ["sparse", "quant", "huffman"],
        HUFFMAN_SIZE_FIG_FILE,
        "Deep Compression step 3: model size after Huffman coding\n"
        f"(dense conv/fc weights = {kb(dense_bits):.0f} KB)",
    )
    return stat_rows


def main():
    parser = argparse.ArgumentParser(description="Deep Compression 后续阶段")
    parser.add_argument(
        "--stage",
        default="sparse",
        choices=["sparse", "quant", "huffman"],
        help="执行哪个阶段",
    )
    parser.add_argument(
        "--sparsities",
        default=",".join(f"{s}" for s in SPARSITIES),
        help="逗号分隔的稀疏度列表",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=QUANT_FINETUNE_EPOCHS,
        help="centroid 微调的 epoch 数（仅 --stage quant）",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="只量化，不做 centroid 微调（仅 --stage quant）",
    )
    args = parser.parse_args()

    sparsities = [float(s) for s in args.sparsities.split(",")]
    print(f"设备：{device}  阶段：{args.stage}  稀疏度：{sparsities}")
    trainloader, testloader = build_loaders()
    criterion = torch.nn.CrossEntropyLoss()

    if args.stage == "sparse":
        run_sparse(testloader, criterion, sparsities)
    elif args.stage == "quant":
        run_quant(
            testloader,
            trainloader,
            criterion,
            sparsities,
            args.epochs,
            skip_finetune=args.skip_finetune,
        )
    elif args.stage == "huffman":
        run_huffman(testloader, criterion, sparsities)


if __name__ == "__main__":
    main()
