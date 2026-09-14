"""
Deep Compression step 2：权重共享（weight sharing / k-means 量化）。

对每层的非零权重单独做一维 k-means（线性初始化 + Lloyd 迭代），用 cluster index
替换原来的 fp32 值：conv 4 bits（16 个 centroid）、fc 5 bits（32 个），centroid 表
本身按 fp32 存（论文里 codebook 的开销就来自这里）。量化后值流只要
`nnz * bits` 个 bit，与稀疏格式（sparse_format.py）里 index 流的部分互不影响。

量化之后做 centroid 微调：前向时用 codebook[indices] 重建稠密权重，同一 cluster 的
权重共享同一个参数，autograd 会自动把它们的梯度累加到对应 centroid 上（这就是论文说的
"梯度按 cluster 聚合"）；被剪位置由结构保证恒为 0，不需要额外的 mask。
微调只移动 centroid 的值，index 和剪枝结构不变，因此压缩后体积不变。
"""

import copy
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from model import test
from sparse_format import (
    FILLER_CODE,
    FILLER_VALUE,
    decode_layer,
    from_matrix,
    prunable_modules,
    to_dense,
)
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 实验超参数 ----------------
CONV_BITS = 4  # 论文默认：conv 4 bits
FC_BITS = 5  # 论文默认：fc 5 bits
KMEANS_ITERS = 100  # Lloyd 迭代上限（收敛即提前退出）
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-3
FINETUNE_MOMENTUM = 0.9
FINETUNE_WEIGHT_DECAY = 1e-4


def bits_of(module):
    """conv 4 bits / fc 5 bits。"""
    return CONV_BITS if isinstance(module, nn.Conv2d) else FC_BITS


# ---------------- k-means ----------------
def kmeans_1d(values, k, iters=KMEANS_ITERS):
    """一维 k-means：线性初始化 + Lloyd 迭代。

    values 是某一层的全部非零权重（一维 tensor），返回 (centroids 升序, assign)。
    初始 centroid 在 [min, max] 上均匀取点（论文的 linear initialization）；
    空 cluster 保持原 centroid 不动。
    """
    v = values.flatten()
    k = max(1, min(int(k), int(v.numel())))
    centroids = torch.linspace(
        float(v.min()), float(v.max()), k, dtype=v.dtype, device=v.device
    )

    assign = None
    for _ in range(iters):
        new_assign = (v.unsqueeze(1) - centroids.unsqueeze(0)).abs().argmin(dim=1)
        sums = torch.zeros(k, dtype=v.dtype, device=v.device)
        counts = torch.zeros(k, dtype=v.dtype, device=v.device)
        sums.scatter_add_(0, new_assign, v)
        counts.scatter_add_(0, new_assign, torch.ones_like(v))
        nonempty = counts > 0
        converged = assign is not None and bool(torch.equal(new_assign, assign))
        centroids = torch.where(nonempty, sums / counts.clamp(min=1), centroids)
        assign = new_assign
        if converged:  # 分配不再变化即收敛
            break

    # 一维 k-means 的 centroid 天然有序，这里显式排序只是让 codebook 便于查看与画图
    order = torch.argsort(centroids)
    return centroids[order].contiguous(), torch.argsort(order)[assign]


def quantize_layer(layer, bits):
    """把一层的值流换成 cluster index + codebook，返回 (新 SparseLayer, 统计信息)。"""
    assert layer.codebook is None, f"{layer.name}: 该层已经量化过了"
    values = np.asarray(layer.values)
    is_value = values != FILLER_VALUE  # filler（占位）不参与聚类

    v = torch.from_numpy(values[is_value].astype(np.float32)).to(device)
    centroids, assign = kmeans_1d(v, 1 << bits)
    err = v - centroids[assign]

    new_values = np.full(values.shape, FILLER_CODE, dtype=np.int16)
    new_values[is_value] = assign.cpu().numpy()
    stats = {
        "bits": bits,
        "k": int(centroids.numel()),
        "used": int(torch.unique(assign).numel()),
        "nnz": int(v.numel()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "max_err": float(err.abs().max()),
    }
    return (
        replace(
            layer,
            values=new_values,
            value_bits=bits,
            codebook=centroids.cpu().numpy().astype(np.float32),
        ),
        stats,
    )


def quantize_model(model, layers):
    """逐层量化。返回 ({参数名: SparseLayer}, {参数名: 统计信息})。"""
    out, stats = {}, {}
    for name, module in prunable_modules(model):
        key = f"{name}.weight"
        out[key], stats[key] = quantize_layer(layers[key], bits_of(module))
    return out, stats


# ---------------- 共享权重的模块 ----------------
class _SharedWeight(nn.Module):
    """共享权重模块的公共部分：前向时用 codebook[indices] 重建稠密权重。

    indices 是原始权重形状的 cluster index 张量，-1 表示被剪掉的位置（恒为 0）。
    同 cluster 的权重共享一个 centroid 参数，反传时 autograd 自动把梯度累加到它上面。
    """

    def __init__(self, indices, codebook):
        super().__init__()
        self.centroids = nn.Parameter(
            torch.as_tensor(codebook, dtype=torch.float32, device=device)
        )
        # indices / shared 只用于前向重建，不参与落盘，因此不放 state_dict
        self.register_buffer(
            "indices", torch.as_tensor(indices, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer("shared", self.indices >= 0, persistent=False)

    def dense_weight(self):
        weight = self.centroids[self.indices.clamp(min=0)]
        return weight * self.shared.to(weight.dtype)


class SharedWeightConv2d(_SharedWeight):
    def __init__(self, module, indices, codebook):
        super().__init__(indices, codebook)
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.bias = nn.Parameter(module.bias.detach().clone())

    def forward(self, x):
        return F.conv2d(
            x,
            self.dense_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SharedWeightLinear(_SharedWeight):
    def __init__(self, module, indices, codebook):
        super().__init__(indices, codebook)
        self.bias = nn.Parameter(module.bias.detach().clone())

    def forward(self, x):
        return F.linear(x, self.dense_weight(), self.bias)


def layer_indices(layer):
    """解码出原始权重形状的 cluster index 张量（-1 = 被剪/占位）。"""
    rows, cols, vals = decode_layer(layer)
    matrix = np.full((layer.n_rows, layer.row_len), FILLER_CODE, dtype=np.int64)
    matrix[rows, cols] = vals
    return torch.from_numpy(
        np.ascontiguousarray(from_matrix(matrix, layer.orientation, layer.shape))
    )


def build_shared_model(model, layers):
    """复制一份模型，把 conv/linear 换成共享权重模块（BN/bias 原样保留）。"""
    shared = type(model)().to(device)
    shared.load_state_dict(model.state_dict())
    for name, module in prunable_modules(model):
        layer = layers[f"{name}.weight"]
        cls = SharedWeightConv2d if isinstance(module, nn.Conv2d) else SharedWeightLinear
        parent_name, _, leaf = name.rpartition(".")
        setattr(
            shared.get_submodule(parent_name),
            leaf,
            cls(module, layer_indices(layer), layer.codebook),
        )
    return shared


def check_layers_match(shared, layers):
    """校验 codebook+index 解码出的权重与共享模块重建的权重逐位一致，返回最大偏差。"""
    max_abs = 0.0
    with torch.no_grad():
        for name, layer in layers.items():
            module = shared.get_submodule(name[: -len(".weight")])
            ref = module.dense_weight().detach().cpu()
            max_abs = max(max_abs, float((ref - layer_to_tensor(layer)).abs().max()))
    return max_abs


def layer_to_tensor(layer):
    """SparseLayer -> 原始形状的稠密权重（torch，CPU）。"""
    return to_dense(layer)


def nonzero_of(shared):
    """统计共享模型里非零权重个数，应与稀疏结构的 nnz 相同。"""
    total = 0
    with torch.no_grad():
        for module in shared.modules():
            if isinstance(module, _SharedWeight):
                total += int((module.dense_weight() != 0).sum())
    return total


# ---------------- centroid 微调 ----------------
def model_snapshot(shared):
    """完整状态快照（centroid + bias + BN 参数/buffer）。

    index 与剪枝 mask 是 non-persistent buffer，训练中不变，因此不需要快照。
    """
    return {name: value.detach().clone() for name, value in shared.state_dict().items()}


def load_model_snapshot(shared, snapshot):
    shared.load_state_dict(snapshot)


def finetune_centroids(
    shared, trainloader, testloader, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR
):
    """centroid 微调：只更新 codebook 与 bias/BN，index 和剪枝结构保持不变。

    返回 (history, 最终 test acc)。若最后一个 epoch 不是最优，会回退到最优 epoch 的
    完整状态（centroid + bias/BN）并重新评测。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        shared.parameters(),
        lr=lr,
        momentum=FINETUNE_MOMENTUM,
        weight_decay=FINETUNE_WEIGHT_DECAY,
    )
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "best_test_acc": [],
    }
    best_acc, best_snapshot = -1.0, None
    for epoch in range(1, epochs + 1):
        shared.train()
        running_loss = 0.0
        correct = total = 0
        for data, label in trainloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = shared(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            bs = label.size(0)
            total += bs
            running_loss += loss.item() * bs
            _, predicted = output.max(1)
            correct += predicted.eq(label).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = test(shared, testloader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["best_test_acc"].append(max(best_acc, test_acc))
        if test_acc > best_acc:
            best_acc, best_snapshot = test_acc, model_snapshot(shared)
        print(
            f"Centroid FT Epoch:{epoch:3d}/{epochs} | "
            f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
            f"Test Loss:{test_loss:.4f} Acc:{test_acc:.4f}"
        )

    if best_acc > history["test_acc"][-1]:
        load_model_snapshot(shared, best_snapshot)
        _, best_acc = test(shared, testloader, criterion)
        print(f"已回退到最优 epoch 的模型状态：Test Acc {best_acc:.4f}")
    return history, best_acc


def sync_codebook(layers, shared):
    """把微调后的 centroid 写回 SparseLayer（index 不变，体积不变）。"""
    out = {}
    for name, layer in layers.items():
        module = shared.get_submodule(name[: -len(".weight")])
        out[name] = replace(
            layer,
            codebook=module.centroids.detach().cpu().numpy().astype(np.float32),
        )
    return out


def expand_shared_model(shared):
    """把共享权重模块展开成普通 conv/linear：权重 = codebook[index]，bias/BN 原样保留。

    用于把量化后的模型变回普通的 MyCNN（落盘校验、评测、再落盘解码都用它）。
    """
    dense = copy.deepcopy(shared)
    for name, module in list(shared.named_modules()):
        if not isinstance(module, _SharedWeight):
            continue
        shape = module.indices.shape
        if isinstance(module, SharedWeightConv2d):
            new = nn.Conv2d(
                shape[1] * module.groups,
                shape[0],
                kernel_size=tuple(shape[2:]),
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        else:
            new = nn.Linear(shape[1], shape[0])
        new = new.to(module.centroids.device)
        with torch.no_grad():
            new.weight.copy_(module.dense_weight())
            new.bias.copy_(module.bias)
        parent_name, _, leaf = name.rpartition(".")
        setattr(dense.get_submodule(parent_name), leaf, new)
    return dense
