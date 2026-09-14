"""
第三步：Huffman 编码（论文里权重 index 与稀疏 index 都过了熵编码）。

- 每层的**值流**（cluster index）和 **index 差值流** 各自独立统计频率、建 Huffman 树；
- 由码长生成规范码（canonical code）：落盘时只需要存每个符号的码长，省掉一张码表；
- 位流 MSB-first 打包，与 sparse_format.pack_bits 的约定一致（每字节从高位往低位填）；
- filler：量化后 filler 的值是保留符号 -1（无法直接当索引），因此编码时统一加 offset=1，
  解码时减回来；本实验里 filler 个数为 0，所以 offset 实际是 0，码长表长度就等于 k；
- 行计数（row_nnz，16 bits）与 codebook（k x fp32）不参与 Huffman，与论文一致。

压缩后文件的体积 = 行计数 + 两张码长表 + 两条码流 + codebook，外加不压缩的 bias/BN。
"""

import heapq
from dataclasses import dataclass

import numpy as np
import torch

from sparse_format import ROW_NNZ_BITS, SparseLayer


# ---------------- 熵编码核心 ----------------
def symbol_counts(symbols, n_symbols):
    """统计各符号出现次数（symbols 必须是非负整数）。"""
    return np.bincount(np.asarray(symbols, dtype=np.int64), minlength=n_symbols)[
        :n_symbols
    ].astype(np.int64)


def entropy_bits(counts):
    """频率分布的熵（bit/符号），用于和 Huffman 实际码长对比。"""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def code_lengths(counts):
    """最小堆建 Huffman 树，返回每个符号的码长（未出现的符号为 0）。

    相同频率时用入堆序号 tie-break，保证结果可复现。
    """
    lengths = np.zeros(len(counts), dtype=np.int64)
    heap = [(int(c), i, [i]) for i, c in enumerate(counts) if c > 0]
    if not heap:
        return lengths
    if len(heap) == 1:  # 只有一种符号：给它 1 bit（没有前缀冲突问题）
        lengths[heap[0][2][0]] = 1
        return lengths

    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        merged = s1 + s2
        for s in merged:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, serial, merged))
        serial += 1
    return lengths


def canonical_codes(lengths):
    """码长 -> 规范码（按码长升序、同码长按符号升序依次分配）。"""
    codes = np.zeros(len(lengths), dtype=np.int64)
    used = [int(s) for s in np.nonzero(lengths)[0]]
    used.sort(key=lambda s: (int(lengths[s]), s))
    code, prev_len = 0, 0
    for s in used:
        length = int(lengths[s])
        code <<= length - prev_len
        codes[s] = code
        code += 1
        prev_len = length
    return codes


def pack_code_stream(symbols, lengths, codes):
    """按规范码把符号序列打包成位流，返回 (bytes, 有效 bit 数)。"""
    out = bytearray()
    acc = n = 0
    for v in symbols:
        length = int(lengths[v])
        acc = (acc << length) | int(codes[v])
        n += length
        while n >= 8:
            n -= 8
            out.append((acc >> n) & 0xFF)
        acc &= (1 << n) - 1
    if n:
        out.append((acc << (8 - n)) & 0xFF)
    return bytes(out), sum(int(lengths[v]) for v in symbols)


def unpack_code_stream(data, lengths, codes, count):
    """位流解码回符号序列（按位走 Huffman 树）。"""
    if count == 0:
        return np.empty(0, dtype=np.int64)

    root = [None, None, None]  # [左, 右, 叶子的符号]
    for sym in np.nonzero(lengths)[0]:
        length, code = int(lengths[sym]), int(codes[sym])
        node = root
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            if node[bit] is None:
                node[bit] = [None, None, None]
            node = node[bit]
        node[2] = int(sym)

    values = np.empty(count, dtype=np.int64)
    node, idx = root, 0
    for byte in data:
        for shift in range(7, -1, -1):
            node = node[(byte >> shift) & 1]
            sym = node[2]
            if sym is not None:
                values[idx] = sym
                idx += 1
                if idx == count:
                    return values
                node = root
    raise AssertionError(f"码流长度不足：只解出 {idx}/{count} 个符号")


@dataclass
class HuffmanStream:
    """一条流的编码结果：码长表 + 位流（码表只有码长，没有码字）。"""

    lengths: np.ndarray  # (n_symbols,) 每符号码长，uint8
    bits: bytes  # 位流
    n_symbols: int  # 符号个数（真实值，不含 offset）
    n_entries: int  # 被编码的符号个数
    n_bits: int  # 位流的有效 bit 数（不含末尾补齐）
    offset: int  # 符号偏移（有 filler 时是 1，把 -1 映射到 0）

    def size_breakdown(self):
        """返回 (码流 bit, 码长表 bit)。"""
        return self.n_bits, int(self.lengths.size) * 8

    def size_bits(self):
        return sum(self.size_breakdown())

    @property
    def bits_per_symbol(self):
        return 0.0 if self.n_entries == 0 else self.n_bits / self.n_entries


def encode_stream(values, n_symbols, offset=0):
    """把整数流编码成 HuffmanStream。"""
    symbols = np.asarray(values, dtype=np.int64).ravel() + offset
    n_alphabet = n_symbols + offset
    assert symbols.size == 0 or int(symbols.max()) < n_alphabet, (
        f"符号越界：最大 {int(symbols.max())} >= 字母表 {n_alphabet}"
    )
    counts = symbol_counts(symbols, n_alphabet)
    lengths = code_lengths(counts)
    codes = canonical_codes(lengths)
    bits, n_bits = pack_code_stream(symbols, lengths, codes)
    return HuffmanStream(
        lengths=lengths.astype(np.uint8),
        bits=bits,
        n_symbols=n_symbols,
        n_entries=int(symbols.size),
        n_bits=int(n_bits),
        offset=offset,
    )


def decode_stream(stream):
    """HuffmanStream -> 原始整数流。"""
    lengths = stream.lengths.astype(np.int64)
    codes = canonical_codes(lengths)
    symbols = unpack_code_stream(stream.bits, lengths, codes, stream.n_entries)
    return symbols - stream.offset


# ---------------- 层 / 模型级别 ----------------
def _value_offset(layer):
    """值流需要偏移 1 才能编码 filler（-1）；没有 filler 就不偏移。"""
    return 1 if layer.n_filler else 0


def layer_streams(layer):
    """一层 -> {"values": HuffmanStream, "diff": HuffmanStream}。"""
    assert layer.codebook is not None, f"{layer.name}: 值流未量化（fp32），无法做 Huffman"
    offset = _value_offset(layer)
    return {
        "values": encode_stream(layer.values, layer.codebook.size, offset),
        "diff": encode_stream(layer.diff, 1 << layer.index_bits, 0),
    }


def layer_size_breakdown(layer, streams):
    """Huffman 之后各段占用（bit）。"""
    value_bits, value_table = streams["values"].size_breakdown()
    diff_bits, diff_table = streams["diff"].size_breakdown()
    return {
        "row_nnz": layer.n_rows * ROW_NNZ_BITS,
        "index": diff_bits,
        "index_table": diff_table,
        "value": value_bits,
        "value_table": value_table,
        "codebook": 0 if layer.codebook is None else layer.codebook.size * 32,
    }


def encode_huffman(layers):
    """{name: SparseLayer} -> {name: streams}。"""
    return {name: layer_streams(layer) for name, layer in layers.items()}


def huffman_size(layers, streams):
    """返回 (总 bit, 各段 bit 合计)，分段比 step 2 多了两张码长表。"""
    total = {
        "row_nnz": 0,
        "index": 0,
        "index_table": 0,
        "value": 0,
        "value_table": 0,
        "codebook": 0,
    }
    for name, layer in layers.items():
        for k, v in layer_size_breakdown(layer, streams[name]).items():
            total[k] += v
    return sum(total.values()), total


def restore_layers(layers, streams):
    """用码流解码出新的 SparseLayer 集合（值/差值流与原层逐位一致）。"""
    out = {}
    for name, layer in layers.items():
        out[name] = SparseLayer(
            name=layer.name,
            shape=layer.shape,
            orientation=layer.orientation,
            row_len=layer.row_len,
            index_bits=layer.index_bits,
            row_nnz=layer.row_nnz.copy(),
            diff=decode_stream(streams[name]["diff"]),
            values=decode_stream(streams[name]["values"]),
            value_bits=layer.value_bits,
            codebook=None if layer.codebook is None else layer.codebook.copy(),
        )
    return out


# ---------------- 落盘 ----------------
def _bytes_tensor(data):
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


def _stream_payload(stream):
    return {
        "lengths": torch.from_numpy(stream.lengths.astype(np.uint8)),
        "bits": _bytes_tensor(stream.bits),
        "n_symbols": stream.n_symbols,
        "n_entries": stream.n_entries,
        "n_bits": stream.n_bits,
        "offset": stream.offset,
    }


def _stream_from_payload(d):
    return HuffmanStream(
        lengths=d["lengths"].numpy().astype(np.int64),
        bits=d["bits"].numpy().tobytes(),
        n_symbols=d["n_symbols"],
        n_entries=d["n_entries"],
        n_bits=d["n_bits"],
        offset=d["offset"],
    )


def save_compressed(path, layers, streams, state, meta=None):
    """落盘：压缩后的权重（Huffman 码流）+ 不压缩的 bias/BN 参数，文件自包含。"""
    payload = {"meta": dict(meta or {}), "state": dict(state), "layers": {}}
    for name, layer in layers.items():
        payload["layers"][name] = {
            "name": layer.name,
            "shape": list(layer.shape),
            "orientation": layer.orientation,
            "row_len": layer.row_len,
            "index_bits": layer.index_bits,
            "value_bits": layer.value_bits,
            "row_nnz": torch.from_numpy(layer.row_nnz.astype(np.uint16)),
            "codebook": (
                None
                if layer.codebook is None
                else torch.from_numpy(np.asarray(layer.codebook, dtype=np.float32))
            ),
            "streams": {k: _stream_payload(s) for k, s in streams[name].items()},
        }
    torch.save(payload, path)


def load_compressed(path):
    """读取压缩文件，返回 (meta, 不压缩部分的状态, 解码后的 SparseLayer 集合)。"""
    payload = torch.load(path, weights_only=True)
    layers = {}
    for name, d in payload["layers"].items():
        codebook = d["codebook"]
        layers[name] = SparseLayer(
            name=d["name"],
            shape=tuple(d["shape"]),
            orientation=d["orientation"],
            row_len=d["row_len"],
            index_bits=d["index_bits"],
            row_nnz=d["row_nnz"].numpy().astype(np.int64),
            diff=decode_stream(_stream_from_payload(d["streams"]["diff"])),
            values=decode_stream(_stream_from_payload(d["streams"]["values"])),
            value_bits=d["value_bits"],
            codebook=None if codebook is None else codebook.numpy(),
        )
    return payload["meta"], payload["state"], layers
