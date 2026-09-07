"""
一个参考AlexNet、针对CIFAR-10调整的CNN网络
总参数 413962
没有做超参数的调优
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_PATH = "D:\\dataset"
LOG_FILE_NAME = "log_baseline.txt"
MODEL_FILE_NAME = "CIFAR10_baseline.pth"

# 固定种子、设备
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 64
NUM_WORKERS = 4  # Windows 下 DataLoader 多进程必须放在 __main__ 保护内创建
LEARNING_RATE = 0.001
MAX_LR = 0.01  # one cycle 调度器
EPOCHS = 100

# CIFAR-10 数据集的均值和标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# 类别名称
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# 测试集和训练集的数据增强需要分开
train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  # tensor shape (3,32,32) CHW
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        # 随机擦除：必须放在 Normalize 之后，value=0 在归一化空间中恰好是"均值像素"
        transforms.RandomErasing(p=0.5),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # 这个均值和标准差是全局的
    ]
)

# 数据集和 DataLoader 的创建移到了 main() 内（见下方注释说明原因）


# 模型
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 两层conv+norm+pool
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),  # (N,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # (N,32,16,16)
            nn.Conv2d(32, 96, kernel_size=3, padding=1, stride=1),  # (N,96,16,16)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # (N,96,8,8)
            # 连续三个卷积层+一个池化层
            nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # (N,128,8,8)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # (N,128,8,8)
            nn.Conv2d(128, 96, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),  # (N,96,8,8)
            nn.MaxPool2d(2, stride=2),  # (N,96,4,4)
            # 线性层，这里改为只用一个
            nn.Flatten(),
            nn.Linear(1536, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def train_epoch(model, trainloader, criterion, optimizer, scheduler):
    """训练一个epoch，返回avg loss和acc"""
    current_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)  # output tensor shape (N,10)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()  # OneCycleLR 以 batch 为单位步进，必须每个 batch 调用一次

        batch_size = label.size(0)
        total += batch_size
        # loss.item() 是 batch 内平均 loss，乘 batch_size 恢复总和
        current_loss += loss.item() * batch_size

        # predicted tensor shape (N)
        # label tensor shape (N)
        _, predicted = output.max(1)  # 按照dim=1取max，输出(value,indices)

        # bool tensor -> tensor shape (1) -> int
        correct += predicted.eq(label).sum().item()

    return current_loss / total, correct / total


def test(model, testloader, criterion):
    """测试整个测试集，返回avg loss和acc"""
    current_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            batch_size = label.size(0)
            total += batch_size
            output = model(data)
            # loss.item() 是 batch 内平均 loss，乘 batch_size 恢复总和
            current_loss += criterion(output, label).item() * batch_size
            _, predicted = output.max(1)
            correct += predicted.eq(label).sum().item()

    return current_loss / total, correct / total


def main():
    print(f"设备：{device}")

    # Windows 的多进程采用 spawn 方式启动，子进程会重新 import 本模块，
    # 若 DataLoader 在模块级创建，会在每个 worker 中重复执行（浪费内存甚至递归报错），
    # 因此必须放在 if __name__ == "__main__" 保护链路内创建
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
        persistent_workers=NUM_WORKERS > 0,  # 避免每个 epoch 重新 spawn worker
    )
    # 测试集没必要shuffle
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"训练集数量：{len(trainset)}")
    print(f"测试集数量：{len(testset)}")

    model = MyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=len(trainloader), epochs=EPOCHS
    )
    print("模型结构：")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params}")

    # 训练
    train_loss, train_acc, train_lr = [], [], []
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(model, trainloader, criterion, optimizer, scheduler)
        train_loss.append(loss)
        train_acc.append(acc)
        train_lr.append(scheduler.get_last_lr()[0])

        print(
            f"Epoch:{epoch:3d}/{EPOCHS} | Loss:{loss:.4f} | Acc:{acc:.4f} | LR:{scheduler.get_last_lr()[0]:.6f}"
        )

    # 保存模型
    torch.save(model.state_dict(), MODEL_FILE_NAME)
    print(f"Model Saved: {MODEL_FILE_NAME}")

    # 测试
    test_loss, test_acc = test(model, testloader, criterion)
    print(f"\n测试集  Loss:{test_loss:.4f}  Acc:{test_acc:.4f}")

    # 写入日志
    with open(LOG_FILE_NAME, "w") as f:
        f.write(f"test Loss:{test_loss:.4f}  Acc:{test_acc:.4f}\n\n")
        for epoch, (loss, acc, lr) in enumerate(
            zip(train_loss, train_acc, train_lr), start=1
        ):
            f.write(
                f"Epoch:{epoch:3d}/{EPOCHS} | Loss:{loss:.4f} | Acc:{acc:.4f} | LR:{lr:.6f}\n"
            )
    print(f"Results Saved: {LOG_FILE_NAME}")


if __name__ == "__main__":
    main()
