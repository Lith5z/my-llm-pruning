# my-llm-pruning

尝试复现以下论文中的剪枝操作

- Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding


受限于算力，会使用以下模型和数据集

- 从AlexNet修改而来的CNN网络（总参数量 413962） + CIFAR10数据集
- Qwen2.5-3B-Instruct + MATH500数据集

前者我能够快速本地训练和测试，后者我勉强能在本地完成推理

## Deep Compression

### 方法

- 常规训练
- 每层对于小于某个门槛threshold的权重，直接置为0，得到一个稀疏的网络
- 再对这个稀疏网络进行微调（使用掩码mask仅对剩余权重进行梯度下降）

对于剪枝后的稀疏网络：

- CSR/CSC格式存储
- 对于CSR下的col_idx数组（或者CSC下是row_idx），再使用index diff储存方式进一步压缩，对于conv层仅需8bits/fc层5bits来存储索引
  - 如果diff超出上限，填充0

## 代码

## 实验

baseline 89.58