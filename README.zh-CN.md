# HybridQCCNN

HybridQCCNN 是一个专为图像分类设计的混合量子-经典卷积神经网络框架。它集成了多种预定义模型，支持不同的量子模拟后端，并提供丰富的可视化与分析工具，以便在经典与量子增强的训练模式下顺利开展工作。

## 目录

- [概述](#概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [常用命令](#常用命令)
- [开发者指南](#开发者指南)
- [项目结构](#项目结构)
- [配置文件](#配置文件)
- [自定义](#自定义)
- [许可证](#许可证)
- [贡献](#贡献)
- [引用](#引用)

## 概述

1. 同时支持经典 CNN 与混合量子-经典 CNN 的统一训练和测试流程
2. 多种预定义模型（如 VGG、GoogLeNet、ResNet）既提供经典版本也提供量子混合版本
3. 支持自定义数据集和模型架构
4. 提供完整的训练可视化和模型性能对比
5. 支持多种量子模拟器（如 PennyLane、Qiskit）

## 安装

### 环境要求

- Python >= 3.10
- （可选）CUDA >= 12.1，用于 GPU 加速
- 操作系统：Linux（推荐）、Windows、macOS

### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/BaizeXS/HybridQCCNN.git
   cd HybridQCCNN
   ```

2. **创建并激活环境**

   ```bash
   conda create -n qml python=3.10
   conda activate qml
   ```

3. **安装依赖**

   ```bash
   pip install -e ".[all-dev]"
   ```

4. **验证安装**

   ```bash
   python -m src.main --help
   ```

### 主要依赖版本

- pennylane ~= 0.40.0
- torch ~= 2.5.1
- torchvision ~= 0.20.1
- numpy ~= 2.0.2
- pandas ~= 2.2.3
- scikit-learn ~= 1.6.0
- matplotlib ~= 3.10.0
- tensorboard >= 2.18.0
- qutip ~= 5.1.1

## 快速开始

以下示例展示如何使用 `configs/benchmark/hybrid_fashionmnist.yaml` 训练并测试一个混合量子-经典模型：

1. **编辑配置文件**

   根据需要调整批次大小、学习率等参数。

2. **训练模型**

   ```bash
   python -m src.main train -c configs/benchmark/hybrid_fashionmnist.yaml
   ```

3. **测试模型**

   ```bash
   python -m src.main test -c configs/benchmark/hybrid_fashionmnist.yaml \
       -w outputs/HybridNet/weights/best_model.pt
   ```

4. **单张图片预测**

   ```bash
   python -m src.main predict -c configs/benchmark/hybrid_fashionmnist.yaml \
       -w outputs/hybrid_fashionmnist/weights.pth -i path/to/image.jpg
   ```

## 常用命令

### 训练

```bash
python -m src.main train -c <CONFIG> [--checkpoint <CHECKPOINT_PATH>]
```

### 测试

```bash
python -m src.main test -c <CONFIG> -w <WEIGHTS_PATH> [--is-checkpoint]
```

### 单张图片预测

```bash
python -m src.main predict -c <CONFIG> -w <WEIGHTS_PATH> -i <IMAGE_PATH> [--is-checkpoint]
```

### 训练指标可视化

```bash
python -m src.main viz-metrics -f <METRICS_FILE> [--metric-names <NAMES>] \
    [--phases <PHASES>] [-o <OUTPUT_DIR>] [--no-show]
```

### 多模型性能对比

```bash
python -m src.main compare -f <METRICS_FILES>... [--metric-names <NAMES>] \
    [--model-names <NAMES>] [--phases <PHASES>] [-o <OUTPUT_DIR>] [--no-show]
```

## 开发者指南

安装开发工具并设置 pre-commit 钩子：

```bash
pip install -e ".[dev]"
pre-commit install
```

### 代码规范

- **black**：代码格式化（默认行宽 88）
- **isort**：import 语句排序
- **flake8**：代码风格检查
- **pre-commit**：自动执行上述检查

运行测试：

```bash
pytest          # 运行所有测试
pytest -m unit  # 仅运行单元测试
```

常用测试标记包括：`slow`, `gpu`, `quantum`, `integration`, `unit`, `heavy_model` 等。

## 项目结构

```
HybridQCCNN/
├── src/                # 源代码（模型、训练脚本、工具函数等）
├── tests/              # 测试文件
├── configs/            # YAML 配置文件
├── examples/           # 示例脚本
├── docs/               # 文档
└── datasets/           # 数据集目录
```

## 配置文件

以下是一个典型的 YAML 配置示例：

```yaml
data:
  name: "DatasetName"
  dataset_type: "DatasetType"
  input_shape: [channels, height, width]
  num_classes: 10
  dataset_path: "path/to/dataset"
  train_split: 0.8
  batch_size: 64

model:
  name: "ModelName"
  model_type: "classic"  # 或 "hybrid"
  quantum_config:
    q_layers: 2
    diff_method: "backprop"
    q_device: "default.qubit"

training:
  learning_rate: 0.001
  weight_decay: 1e-4
  num_epochs: 20

device: "cpu"  # 或 "cuda", "mps"
seed: 42
output_dir: "outputs/experiment_name"
```

对于不同的模型或数据集，可以在 `configs/` 文件夹中分别创建 YAML 文件。

## 自定义

### 添加新数据集

要添加新数据集：

1. 在 `src/datasets/my_dataset.py` 中创建一个继承自 `torch.utils.data.Dataset` 的数据集类：

   ```python
   from utils.data_management import CustomDataset

   class MyDataset(CustomDataset):
       def _load_data(self):
           # 实现数据加载逻辑
           pass

       def __getitem__(self, index):
           # 实现数据检索逻辑
           pass
   ```

2. 在配置文件中设置数据集参数：

   ```yaml
   data:
     name: "MyDataset"  # 与数据集类名称一致
     dataset_type: "CUSTOM"  # 必须设置为CUSTOM
     input_shape: [3, 224, 224]
     num_classes: 10
     dataset_path: "path/to/dataset"
     custom_dataset_path: "path/to/my_dataset.py"
     train_split: 0.8
     batch_size: 32
     # 其他数据配置...
   ```

### 添加新模型

要添加新模型：

1. 在 `src/models/my_model.py` 中创建一个继承自 `torch.nn.Module` 的模型类：

   ```python
   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           # 实现模型结构

       def forward(self, x):
           # 实现前向传播
           return x
   ```

2. 在配置文件中设置模型参数：

   ```yaml
   model:
     name: "MyModel"  # 与类名称一致
     model_type: "custom"  # 必须设置为custom
     model_kwargs:
       num_classes: 10
       hidden_dim: 128
     custom_model_path: "path/to/my_model.py"
     quantum_config: null  # 如果是量子混合模型需要配置
   ```

更多配置文件示例可参见 `templates/custom_config.yaml`。

## 许可证

本项目基于 [MIT License](LICENSE) 进行许可。您可以自由使用、修改和分发本项目的代码，无论用于商业还是非商业目的。

## 引用

如果本项目对您的研究有所帮助，欢迎引用：

```bibtex
@software{HybridQCCNN2024,
  author = {BaizeXS},
  title = {HybridQCCNN: A Hybrid Quantum-Classical CNN Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BaizeXS/HybridQCCNN}
}
```
