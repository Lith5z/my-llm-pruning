"""
剪枝后稀疏权重的存储格式：CSR/CSC + index diff。

- conv2d 权重 (out, in, kh, kw) 展平为 (out, in*kh*kw)，按行（输出通道）存 CSR，
  index 流是 col_idx 的差值，位宽 8 bits；
- linear 权重 (out, in) 转置后按行（输入特征）存 CSC，index 流是 row_idx 的差值，
  位宽 5 bits（行长为 out_features=10，差值恒小于 2^5，不会溢出）；
- 行首元素存相对位置 0 的差值（即绝对 index），其余存与前一个非零值的差值；
- 差值超过位宽上限时按上限拆成多步，插入 filler 条目占位：未量化时 filler 的值是
  0（CSR/CSC 只存非零值，因此 0 可以唯一标识 filler），量化后是保留符号 -1
  （cluster index 是 0..k-1）。filler 只影响稀疏结构，解码时跳过，不改变权重。
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# ---------------- 格式常量 ----------------
CONV_INDEX_BITS = 8  # conv：col_idx 差值位宽
FC_INDEX_BITS = 5  # fc：row_idx 差值位宽
ROW_NNZ_BITS = 16  # 每行的真实非零个数
FLOAT_VALUE_BITS = 32  # 未量化时值流用 fp32
FILLER_VALUE = 0.0  # 未量化时 filler 在值流中的取值
FILLER_CODE = -1  # 量化后 filler 的保留符号


def is_prunable(module):
    return isinstance(module, (nn.Conv2d, nn.Linear))


def prunable_modules(model):
    """返回 [(模块名, 模块)]，模块名与 pruning.py 里 mask 的 key（name + '.weight'）对应。"""
    return [(name, m) for name, m in model.named_modules() if is_prunable(m)]


def index_bits_of(module):
    """conv 8 bits / fc 5 bits（论文设定）"""
    return CONV_INDEX_BITS if isinstance(module, nn.Conv2d) else FC_INDEX_BITS


def orientation_of(module):
    """conv 按 (out, in*kh*kw) 存 CSR；linear 转置成 (in, out) 存 CSC。"""
    return "csr" if isinstance(module, nn.Conv2d) else "csc"


def to_matrix(weight, orientation):
    """原始权重（numpy）-> 压缩时按行存储的二维矩阵。"""
    weight = np.asarray(weight)
    if orientation == "csr":
        return weight.reshape(weight.shape[0], -1)
    return weight.T.reshape(weight.shape[1], -1)


def from_matrix(matrix, orientation, shape):
    """二维矩阵 -> 原始权重形状（numpy）。"""
    if orientation == "csr":
        return matrix.reshape(shape)
    return matrix.T.reshape(shape)


# ---------------- 数据结构 ----------------
@dataclass
class SparseLayer:
    """一层的稀疏存储。row_nnz / diff / values 三段加上 codebook 就是落盘内容。"""

    name: str  # 参数名，如 "model.0.weight"
    shape: tuple  # 原始权重形状
    orientation: str  # "csr"（行=输出维）/ "csc"（行=输入维，矩阵已转置）
    row_len: int  # 行长度（矩阵列数）
    index_bits: int  # index 差值的位宽
    row_nnz: np.ndarray  # (n_rows,) 每行真实非零个数，不含 filler
    diff: np.ndarray  # 差值流（含 filler 步进），int64
    values: np.ndarray  # 值流，与 diff 等长；fp32 或 cluster index
    value_bits: int  # 值位宽：32 = fp32，否则是 cluster index 位宽
    codebook: np.ndarray | None = None  # 权重共享阶段的 centroid

    @property
    def n_rows(self):
        return int(self.row_nnz.size)

    @property
    def nnz(self):
        return int(self.row_nnz.sum())

    @property
    def n_entries(self):
        """值流长度 = 非零值个数 + filler 个数"""
        return int(self.diff.size)

    @property
    def n_filler(self):
        return self.n_entries - self.nnz

    @property
    def max_diff(self):
        return int(self.diff.max()) if self.diff.size else 0

    def size_breakdown(self):
        """各段占用（bit）。"""
        return {
            "row_nnz": self.n_rows * ROW_NNZ_BITS,
            "index": self.n_entries * self.index_bits,
            "value": self.n_entries * self.value_bits,
            "codebook": 0 if self.codebook is None else self.codebook.size * 32,
        }

    def size_bits(self):
        return sum(self.size_breakdown().values())


# ---------------- 编码 ----------------
def _encode_row(index, values, index_bits):
    """一行：index（行内位置，升序）与对应值 -> (差值流, 值流, filler 个数)。"""
    max_diff = (1 << index_bits) - 1
    diffs, vals = [], []
    prev = 0  # 行首元素存相对位置 0 的差值，即绝对 index
    filler = 0
    for pos, val in zip(index, values):
        pos = int(pos)
        gap = pos - prev
        while gap > max_diff:  # 差值超上限：按上限拆步，插入 filler 占位
            diffs.append(max_diff)
            vals.append(FILLER_VALUE)
            prev += max_diff
            gap = pos - prev
            filler += 1
        diffs.append(gap)
        vals.append(val)
        prev = pos
    return diffs, vals, filler


def encode_layer(name, module):
    """把一层的权重编码成 SparseLayer（值仍为 fp32，量化在权重共享阶段做）。"""
    weight = module.weight.detach().cpu()
    orientation = orientation_of(module)
    matrix = to_matrix(weight.numpy(), orientation)

    index_bits = index_bits_of(module)
    row_nnz = np.zeros(matrix.shape[0], dtype=np.int64)
    diff, values = [], []
    for r in range(matrix.shape[0]):
        cols = np.nonzero(matrix[r])[0]
        row_nnz[r] = cols.size
        if cols.size:
            d, v, _ = _encode_row(cols, matrix[r][cols], index_bits)
            diff.extend(d)
            values.extend(v)

    return SparseLayer(
        name=name,
        shape=tuple(weight.shape),
        orientation=orientation,
        row_len=matrix.shape[1],
        index_bits=index_bits,
        row_nnz=row_nnz,
        diff=np.array(diff, dtype=np.int64),
        values=np.array(values, dtype=np.float32),
        value_bits=FLOAT_VALUE_BITS,
    )


def encode_model(model):
    """编码整个模型的 conv/linear 权重，返回 {参数名: SparseLayer}。"""
    return {f"{name}.weight": encode_layer(name, m) for name, m in prunable_modules(model)}


# ---------------- 解码 ----------------
def decode_layer(layer):
    """解码出 (rows, cols, values) 三个扁平数组。

    rows 是存储方向的行号；cols 是行内位置（csr 为 col_idx，csc 为 row_idx）；
    values 是对应的值（filler 已跳过）。
    """
    values = layer.values
    is_filler = values == FILLER_VALUE if layer.codebook is None else values == FILLER_CODE

    rows, cols, vals = [], [], []
    n_rows = layer.n_rows
    row = 0
    while row < n_rows and layer.row_nnz[row] == 0:  # 空行不占值流，直接跳过
        row += 1
    pos, left = 0, int(layer.row_nnz[row]) if row < n_rows else 0

    for diff, val, filler in zip(layer.diff, values, is_filler):
        pos += int(diff)
        assert pos < layer.row_len, f"{layer.name}: 第 {row} 行位置越界（{pos} >= {layer.row_len}）"
        if filler:
            continue
        rows.append(row)
        cols.append(pos)
        vals.append(val)
        left -= 1
        if left == 0:  # 该行结束，跳到下一个非空行
            row += 1
            while row < n_rows and layer.row_nnz[row] == 0:
                row += 1
            pos, left = 0, int(layer.row_nnz[row]) if row < n_rows else 0

    assert left == 0 and len(rows) == layer.nnz, f"{layer.name}: 值流与 row_nnz 不一致"
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(vals)


def to_matrix_dense(layer):
    """解码成存储方向的稠密矩阵（filler 位置为 0）。"""
    rows, cols, vals = decode_layer(layer)
    if layer.codebook is not None:
        vals = np.asarray(layer.codebook)[vals]
    matrix = np.zeros((layer.n_rows, layer.row_len), dtype=np.float32)
    matrix[rows, cols] = np.asarray(vals, dtype=np.float32)
    return matrix


def to_dense(layer):
    """解码成原始形状的权重张量。"""
    matrix = to_matrix_dense(layer)
    return torch.from_numpy(np.ascontiguousarray(from_matrix(matrix, layer.orientation, layer.shape)))


def apply_to_model(model, layers):
    """把稀疏表示解码回模型权重（原地），用于验证编码是否无损。"""
    with torch.no_grad():
        for name, layer in layers.items():
            module = model.get_submodule(name[: -len(".weight")])
            module.weight.copy_(to_dense(layer).to(module.weight.dtype))


# ---------------- 大小统计 ----------------
def sum_size(layers):
    """返回 (总 bit 数, 各段 bit 数合计)。"""
    total = {"row_nnz": 0, "index": 0, "value": 0, "codebook": 0}
    for layer in layers.values():
        for k, v in layer.size_breakdown().items():
            total[k] += v
    return sum(total.values()), total


def dense_weights_bits(model):
    """conv/linear 权重按 fp32 存储的 bit 数（dense 基线）。"""
    return sum(m.weight.numel() * 32 for _, m in prunable_modules(model))


def other_output_size(model):
    """不参与压缩的参数（bias / BN）与 buffer 的字节数。"""
    prunable = {f"{n}.weight" for n, _ in prunable_modules(model)}
    params = sum(
        p.numel() * p.element_size() for n, p in model.named_parameters() if n not in prunable
    )
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params, buffers


# ---------------- 位流打包 ----------------
def pack_bits(values, bits):
    """定宽位流打包（高位在前），返回 bytes。"""
    values = np.asarray(values, dtype=np.int64).ravel()
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    if bits == 16:
        return values.astype(np.uint16).tobytes()
    if bits == 32:
        return values.astype(np.uint32).tobytes()
    out = bytearray()
    acc, n = 0, 0
    for v in values:
        acc = (acc << bits) | int(v)
        n += bits
        while n >= 8:
            n -= 8
            out.append((acc >> n) & 0xFF)
    if n:
        out.append((acc << (8 - n)) & 0xFF)
    return bytes(out)


def unpack_bits(data, bits, count):
    """pack_bits 的逆操作。"""
    if bits == 8:
        return np.frombuffer(data, dtype=np.uint8, count=count).astype(np.int64)
    if bits == 16:
        return np.frombuffer(data, dtype=np.uint16, count=count).astype(np.int64)
    if bits == 32:
        return np.frombuffer(data, dtype=np.uint32, count=count).astype(np.int64)
    values, acc, n = [], 0, 0
    mask = (1 << bits) - 1
    for byte in data:
        acc = (acc << 8) | int(byte)
        n += 8
        while n >= bits and len(values) < count:
            n -= bits
            values.append((acc >> n) & mask)
    assert len(values) == count, f"位流长度不足：{len(values)} != {count}"
    return np.array(values, dtype=np.int64)


def _bytes_tensor(data):
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


def save_layers(path, layers, meta=None, state=None):
    """把 SparseLayer 集合落盘（位流打包），meta 用于记录准确率等附加信息。

    state 是「不压缩部分」的参数（bias / BN 参数与 buffer）：带上它就得到一个自包含的
    文件，可以只靠文件解码出完整模型。
    """
    payload = {"meta": dict(meta or {}), "layers": {}}
    if state is not None:
        payload["state"] = dict(state)
    for name, layer in layers.items():
        quantized = layer.codebook is not None
        # 量化后是 cluster index，可以按 value_bits 打包成真正的 4/5-bit 位流；
        # 只有在出现 filler（-1 无法用无符号位域表示）时才退回 int16 原样存
        packed = quantized and layer.n_filler == 0
        if packed:
            values = pack_bits(layer.values, layer.value_bits)
        else:
            values = np.asarray(layer.values).astype(
                np.int16 if quantized else np.float32
            ).tobytes()
        payload["layers"][name] = {
            "name": layer.name,
            "shape": list(layer.shape),
            "orientation": layer.orientation,
            "row_len": layer.row_len,
            "index_bits": layer.index_bits,
            "row_nnz": torch.from_numpy(layer.row_nnz.astype(np.uint16)),
            "n_diff": int(layer.diff.size),
            "diff": _bytes_tensor(pack_bits(layer.diff, layer.index_bits)),
            "value_bits": layer.value_bits,
            "values_packed": packed,
            "values": _bytes_tensor(values),
            "codebook": (
                None
                if layer.codebook is None
                else torch.from_numpy(np.asarray(layer.codebook, dtype=np.float32))
            ),
        }
    torch.save(payload, path)


def load_layers(path):
    """读取 save_layers 落盘的内容，还原成 SparseLayer 集合。"""
    payload = torch.load(path, weights_only=True)
    layers = {}
    for name, d in payload["layers"].items():
        codebook = d["codebook"]
        codebook = None if codebook is None else codebook.numpy()
        raw = d["values"].numpy().tobytes()
        if d.get("values_packed"):
            values = unpack_bits(raw, d["value_bits"], d["n_diff"]).astype(np.int16)
        else:
            values = np.frombuffer(
                raw, dtype=np.int16 if codebook is not None else np.float32
            ).copy()
        layers[name] = SparseLayer(
            name=d["name"],
            shape=tuple(d["shape"]),
            orientation=d["orientation"],
            row_len=d["row_len"],
            index_bits=d["index_bits"],
            row_nnz=d["row_nnz"].numpy().astype(np.int64),
            diff=unpack_bits(d["diff"].numpy().tobytes(), d["index_bits"], d["n_diff"]),
            values=values,
            value_bits=d["value_bits"],
            codebook=codebook,
        )
    return payload["meta"], layers


def load_state(path):
    """读取文件里附带的不压缩参数（bias / BN）；老文件没有这一段时返回 None。"""
    return torch.load(path, weights_only=True).get("state")


def kb(bits):
    return bits / 8.0 / 1024.0
