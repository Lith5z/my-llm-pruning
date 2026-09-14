# my-llm-pruning

*使用AI进行代码生成*

尝试复现以下论文中的剪枝操作

1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)【已完成】
2. [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
3. [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

受限于算力，会使用以下模型和数据集

- 从AlexNet修改而来的CNN网络（总参数量 413962） + CIFAR10数据集
- Qwen2.5-3B-Instruct + MATH500数据集

前者我能够快速本地训练和测试，后者我勉强能在本地完成推理

前者相对原版AlexNet的修改
1. LRN -> BatchNorm
2. 每个conv后都跟BN 原版conv3~5之间没有归一化
3. 全部使用3*3卷积核
4. FC层只用单层 只做简单映射到类别数

## Deep Compression

```shell
python model.py
python pruning.py
python compression.py --stage sparse|quant|huffman
```

局限性：

- 没有实现三个步骤，尤其是剪枝和量化的消融实验；
- 超参数的选取比较随意，没有经过筛选，尤其是没有按层敏感度分级稀疏度；
- AI实现的代码未经过审查

### 结论

| sparsity | acc（剪枝） | acc（+权重共享） | acc（+Huffman） | 稀疏 KB | +权重共享 KB | +Huffman KB | 稀疏 | +权重共享 | +Huffman |
| -------- | ----------- | ---------------- | --------------- | ------- | ------------ | ----------- | ---- | --------- | -------- |
| 0.50 | 0.8925 | 0.8908 | 0.8908 | 1008.23 | 304.63 | 148.44 | 1.60x | 5.29x | 10.86x |
| 0.60 | 0.8901 | 0.8860 | 0.8860 | 807.38 | 244.58 | 129.33 | 2.00x | 6.59x | 12.46x |
| 0.70 | 0.8834 | 0.8802 | 0.8802 | 606.52 | 184.53 | 106.07 | 2.66x | 8.73x | 15.19x |
| 0.80 | 0.8639 | 0.8574 | 0.8574 | 405.66 | 124.48 | 79.01 | 3.97x | 12.94x | 20.39x |
| 0.90 | 0.7807 | 0.7779 | 0.7779 | 204.81 | 64.43 | 47.51 | 7.87x | 25.01x | 33.92x |

压缩率和论文35x且基本不损失精度差距有点大，不过实验所用的网络和AlexNet有比较大的不同，主要是大大减少了fc层，96.3%参数在conv，而论文绝大部分在fc层，fc层又是对剪枝最不敏感的

### 文件

```
|   log_baseline.txt
|   log_prune.txt
|   log_compression.txt
|   model.py                    # 训练和评测模型
|   plotting.py                 # 画图相关
|   pruning.py                  # 剪枝、微调和再评测
|   sparse_format.py            # 稀疏存储格式：CSR/CSC + index diff
|   weight_sharing.py           # 权重共享：逐层 k-means 量化 + centroid 微调
|   huffman.py                  # Huffman 编码
|   compression.py              # 剪枝之后步骤的脚本
|   pruning_results.png
|   pruning_size_results.png
|   sparse_size_results.png
|   quant_size_results.png
|   quant_accuracy_results.png
|   weight_distribution.png
|   huffman_size_results.png
```

### 剪枝

- 常规训练
- 每层的权重里（无偏置），对于大小小于该层分位数sparsity的权重，直接置为0，得到一个稀疏的网络（非结构化剪枝）
- 再对这个稀疏网络进行微调（使用掩码mask仅对剩余权重进行梯度下降）

### 剪枝+微调结果

![pruning results](DeepCompression/pruning_results.png)

| sparsity | kept | 剪枝后 acc | 微调后 acc | 非零参数 | 仅存非零值 KB | 压缩比 |
| -------- | ---- | ---------- | ---------- | -------- | ------------- | ------ |
| 0.50 | 0.5000 | 0.7260 | 0.8925 | 207706 | 811.35 | 1.99x |
| 0.60 | 0.4000 | 0.4611 | 0.8901 | 166455 | 650.21 | 2.49x |
| 0.70 | 0.3000 | 0.2232 | 0.8834 | 125205 | 489.08 | 3.31x |
| 0.80 | 0.2000 | 0.1214 | 0.8639 | 83954 | 327.95 | 4.93x |
| 0.90 | 0.1000 | 0.1159 | 0.7807 | 42704 | 166.81 | 9.69x |

baseline acc 为 0.8958（dense 网络测试集准确率）

不微调单纯剪枝几乎是不可行的

![pruning size results](DeepCompression/pruning_size_results.png)

这里的压缩比基本上等于 `1/(1-sparsity)`，因为当时只统计了非零值，没有算索引开销

### CSR+index diff

对于剪枝后的稀疏网络：

- CSR/CSC格式存储
- 对于CSR下的col_idx数组（或者CSC下是row_idx），再使用index diff储存方式进一步压缩，对于conv层仅需8bits/fc层5bits来存储索引
  - 如果diff超出上限，填充0

### 压缩结果

- conv 权重 `(out, in, kh, kw)` 展平成 `(out, in*kh*kw)` 按行存 **CSR**，fc 权重 `(out, in)` 转置后按行存 **CSC**；
  index 流存同一行内的索引差值（行首是相对位置 0 的差值，即绝对索引），conv 8 bits / fc 5 bits
- 差值超过位宽上限时按上限拆成多步并插入 filler 占位（未量化时 filler=0，量化后是保留符号 -1），解码时跳过、不改变权重；
  本实验 filler 实际为 0（conv 最大差值 164 < 255，fc 行内 9 < 31，说明这两个位宽够用）
- 每行另存 16 bits 的真实非零个数，用于区分"一行结束"与"下一行开始"；这一步的值流仍是 fp32

稀疏存储不改变模型权重，因此也不影响准确率

| sparsity | kept | acc | 值流 KB | index KB | 行计数 KB | 合计 KB | 落盘文件 KB | 压缩比 |
| -------- | ---- | --- | ------- | -------- | --------- | ------- | ----------- | ------ |
| 0.50 | 0.5000 | 0.8925 | 805.69 | 198.61 | 3.94 | 1008.23 | 1014.72 | 1.60x |
| 0.60 | 0.4000 | 0.8901 | 644.55 | 158.89 | 3.94 | 807.38 | 813.60 | 2.00x |
| 0.70 | 0.3000 | 0.8834 | 483.42 | 119.17 | 3.94 | 606.52 | 612.79 | 2.66x |
| 0.80 | 0.2000 | 0.8639 | 322.28 | 79.45 | 3.94 | 405.66 | 411.85 | 3.97x |
| 0.90 | 0.1000 | 0.7807 | 161.15 | 39.72 | 3.94 | 204.81 | 210.97 | 7.87x |

索引需要额外开销，最后实现相比只存储非零值剩下79%~80%的压缩率（2.00x→1.60x、5.00x→3.97x、10.00x→7.87x）

（以下各节的压缩比都以 dense conv/fc 权重 1611.38 KB 为分母；bias/BN ≈ 9.45 KB 不压缩，随文件原样保存）

![sparse size results](DeepCompression/sparse_size_results.png)

### 权重共享

对于**每一层剪枝后**的权重矩阵（或者是values向量），通过linear初始化得到本组的centroid作为共享权重，再使用一个低位数的cluster index矩阵即可让原权重矩阵映射到（少得多的）centroids数组（也叫codebook）上

再将梯度也进行相同分组，把每组的累积梯度作为centroid的梯度，对共享权重进行微调

逐层独立进行的，不同层之间不共享权重

### 权重共享结果

- 位宽用论文默认值：conv 4 bits（k=16）、fc 5 bits（k=32）；**逐层独立**做 k-means，输入是该层剪枝后剩下的非零权重，
  centroid 用 min/max 线性初始化，再跑 Lloyd 迭代（上限 100 次，不再移动就提前结束）
- 量化把每个非零权重换成离它最近的 centroid：值流由 fp32 变成 `bits` 位的 cluster index，另存 k 个 fp32 的 codebook
- 微调：`SharedWeightConv2d` / `SharedWeightLinear` 用 `codebook[indices]` 重建稠密权重前向，反传时 autograd 自动把同一 cluster 的
  梯度累加到该 centroid 上（等价于论文的"按组累加梯度更新共享权重"）；**index 与剪枝 mask 全程不变**，bias/BN 照常训练
- 每个 epoch 都存一份完整 state_dict（centroid + bias/BN），最后回退到测试集准确率最高的那个 epoch，避免末轮抖动
- 落盘文件自包含（index 位流 + codebook + 行计数 + bias/BN），可只用 `quant_s*.pt` 重建模型评测

| sparsity | nnz | acc（剪枝） | acc（量化） | acc（微调） | 值流 KB | index KB | codebook KB | 行计数 KB | 合计 KB | 压缩比 |
| -------- | --- | ----------- | ----------- | ----------- | ------- | -------- | ----------- | --------- | ------- | ------ |
| 0.50 | 206256 | 0.8925 | 0.8822 | 0.8908 | 101.65 | 198.61 | 0.438 | 3.94 | 304.63 | 5.29x |
| 0.60 | 165005 | 0.8901 | 0.8821 | 0.8860 | 81.32 | 158.89 | 0.438 | 3.94 | 244.58 | 6.59x |
| 0.70 | 123755 | 0.8834 | 0.8719 | 0.8802 | 60.99 | 119.17 | 0.438 | 3.94 | 184.53 | 8.73x |
| 0.80 | 82504 | 0.8639 | 0.8442 | 0.8574 | 40.66 | 79.45 | 0.438 | 3.94 | 124.48 | 12.94x |
| 0.90 | 41254 | 0.7807 | 0.7569 | 0.7779 | 20.33 | 39.72 | 0.438 | 3.94 | 64.43 | 25.01x |

这一步只改了值流（32 bits → conv 4 / fc 5 bits），index 流与行计数和上一步相同，所以收益随稀疏度放大：
1.60x→5.29x、2.66x→8.73x、7.87x→25.01x；逐层量化 RMSE 0.004~0.038，加权 RMSE 0.0055~0.0084。

![quant size results](DeepCompression/quant_size_results.png)

![quant accuracy results](DeepCompression/quant_accuracy_results.png)

![weight distribution](DeepCompression/weight_distribution.png)

### 霍夫曼编码

经过上述操作之后，权重的大小分布在两个峰值附近，权重diff index的分布高度偏向0的一侧，对两者分别使用huffman coding进一步减小26%~51%的空间

### huffman coding结果

- 每层两条流分别编码：**值流**（cluster index，字母表大小 k）和 **index 差值流**（字母表 `1 << index_bits`，conv 256 / fc 32）
- 最小堆建树，再由码长生成**规范码**（落盘只需存码长表，不用存码表本身）；位流 MSB-first 打包，与 `sparse_format` 的约定一致
- filler 用 `offset=1` 统一偏移，保证符号从 0 开始（本实验 filler 为 0，故 offset 为 0）
- 行计数（16 bits）与 codebook 不参与 Huffman（论文里也没编码它们），两条码长表合计约 1.39 KB

| sparsity | acc | 值流 KB | index 流 KB | 码长表 KB | 行计数+codebook KB | 合计 KB | 压缩比 | 相对上一步 |
| -------- | --- | ------- | ----------- | --------- | ------------------ | ------- | ------ | ----------- |
| 0.50 | 0.8908 | 92.09 | 50.58 | 1.39 | 4.38 | 148.44 | 10.86x | -51.3% |
| 0.60 | 0.8860 | 73.86 | 49.71 | 1.39 | 4.38 | 129.33 | 12.46x | -47.1% |
| 0.70 | 0.8802 | 55.49 | 44.81 | 1.39 | 4.38 | 106.07 | 15.19x | -42.5% |
| 0.80 | 0.8574 | 37.41 | 35.84 | 1.39 | 4.38 | 79.01 | 20.39x | -36.5% |
| 0.90 | 0.7779 | 18.79 | 22.96 | 1.39 | 4.38 | 47.51 | 33.92x | -26.3% |

index 差值流是主要受益者：实测只要 2.01~4.56 bits/符号（定长分别是 8 / 5 bits，省 42%~75%），已经很接近它的熵 2.00~4.51；
值流因为 cluster index 接近均匀分布，整体只能省 8%~9%（3.66~3.73 bits/符号）。

落盘文件 `compressed_s*.pt` 还带 bias/BN 与文件头（例如 0.50 是 178.54 KB）；只用该文件重建模型，权重与上一步逐位相同（最大偏差 0），准确率一位不差。

![huffman size results](DeepCompression/huffman_size_results.png)