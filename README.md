# HybridQCCNN

一个混合量子-经典卷积神经网络(Hybrid Quantum-Classical CNN)的实现。

## 简介

HybridQCCNN是一个结合了量子计算和经典深度学习的框架,用于图像分类任务。该项目实现了以下功能:

1. 支持经典CNN和混合量子-经典CNN的训练与测试
2. 提供多种预定义模型(VGG、GoogLeNet、ResNet的经典版本和混合版本)
3. 支持自定义数据集和模型架构
4. 提供完整的训练过程可视化和模型性能对比工具
5. 支持多种量子模拟器后端

## 环境配置

### 1. 基础环境要求

- Python >= 3.10
- CUDA >= 12.1 (可选,用于GPU加速)
- 操作系统: Linux(推荐)、Windows、macOS

### 2. 安装步骤

#### 2.1 从PyPI安装

```bash
# 基础安装
pip install hybridqccnn

# 安装带后端API的版本
pip install hybridqccnn[backend]

# 安装开发版本(包含测试和开发工具)
pip install hybridqccnn[all-dev]
```

#### 2.2 从源码安装

```bash
# 克隆仓库
git clone https://github.com/BaizeXS/HybridQCCNN.git
cd HybridQCCNN

# 创建并激活conda环境
conda create -n qml python=3.10
conda activate qml

# 安装依赖
pip install -e ".[all-dev]"  # 安装所有依赖(包括开发工具)
```

### 3. 核心依赖版本

- pennylane ~= 0.40.0
- torch ~= 2.5.1
- torchvision ~= 0.20.1
- numpy ~= 2.0.2
- pandas ~= 2.2.3
- scikit-learn ~= 1.6.0
- matplotlib ~= 3.10.0
- tensorboard >= 2.18.0
- qutip ~= 5.1.1

## 使用说明

### 1. 模型训练

```bash
python -m src.main train -c configs/benchmark/hybrid_fashionmnist.yaml
```

支持的参数:
- `-c/--config`: 配置文件路径(必需)
- `--checkpoint`: 检查点文件路径(可选,用于继续训练)

### 2. 模型测试

```bash
python -m src.main test -c configs/benchmark/hybrid_fashionmnist.yaml -w path/to/weights.pth
```

支持的参数:
- `-c/--config`: 配置文件路径(必需)
- `-w/--weights`: 模型权重文件路径(必需)
- `--is-checkpoint`: 指定权重文件是否为检查点文件

### 3. 单张图片预测

```bash
python -m src.main predict -c configs/benchmark/hybrid_fashionmnist.yaml -w path/to/weights.pth -i path/to/image.jpg
```

支持的参数:
- `-c/--config`: 配置文件路径(必需)
- `-w/--weights`: 模型权重文件路径(必需)
- `-i/--image`: 待预测图片路径(必需)
- `--is-checkpoint`: 指定权重文件是否为检查点文件

### 4. 训练指标可视化

```bash
python -m src.main viz-metrics -f path/to/metrics.json
```

支持的参数:
- `-f/--file`: 指标文件路径(必需)
- `--metric-names`: 要可视化的指标名称列表
- `--phases`: 要可视化的训练阶段列表
- `-o/--output-dir`: 图像保存目录
- `--no-show`: 不显示图像

### 5. 多模型性能对比

```bash
python -m src.main compare -f metrics1.json metrics2.json
```

支持的参数:
- `-f/--files`: 多个指标文件路径(必需)
- `--metric-names`: 要对比的指标名称列表
- `--model-names`: 要对比的模型名称列表
- `--phases`: 要对比的训练阶段列表
- `-o/--output-dir`: 图像保存目录
- `--no-show`: 不显示图像

## 开发指南

### 1. 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit hooks
pre-commit install
```

### 2. 代码规范

本项目使用以下工具确保代码质量：

- black: 代码格式化(行长度88字符)
- isort: import语句排序
- flake8: 代码风格检查
- pre-commit: 自动运行以上工具

### 3. 测试

```bash
# 运行所有测试
pytest

# 运行特定类型的测试
pytest -m "not slow"  # 跳过耗时测试
pytest -m "not gpu"   # 跳过需要GPU的测试
pytest -m unit        # 只运行单元测试
```

支持的测试标记：
- slow: 耗时测试
- gpu: 需要GPU的测试
- quantum: 使用量子模拟器的测试
- integration: 集成测试
- unit: 单元测试
- heavy_model: 资源密集型模型测试

### 4. 项目结构

```
HybridQCCNN/
├── src/                # 源代码
├── tests/             # 测试文件
├── configs/           # 配置文件
├── examples/          # 示例代码
├── docs/              # 文档
└── datasets/          # 数据集目录
```

## 配置文件说明

配置文件采用YAML格式,主要包含以下几个部分:

```yaml
# 数据集配置
data:
  name: "数据集名称"
  dataset_type: "数据集类型"
  input_shape: [通道数, 高度, 宽度]
  num_classes: 类别数
  dataset_path: "数据集路径"
  train_split: 训练集比例
  batch_size: 批次大小
  # ...

# 模型配置
model:
  name: "模型名称"
  model_type: "模型类型(classic/hybrid)"
  quantum_config:  # 量子配置(仅hybrid类型需要)
    q_layers: 量子层数
    diff_method: "量子梯度计算方法"
    q_device: "量子设备类型"
    # ...

# 训练配置
training:
  learning_rate: 学习率
  weight_decay: 权重衰减
  num_epochs: 训练轮数
  # ...

# 其他配置
device: "cpu/cuda"
seed: 随机种子
output_dir: "输出目录"
```

## 自定义扩展

### 1. 添加新的数据集

1. 在`src/datasets`目录下创建新的数据集类
2. 继承`torch.utils.data.Dataset`
3. 在`src/utils/data_management.py`中注册新数据集

### 2. 添加新的模型架构

1. 在`src/models`目录下创建新的模型类
2. 继承`torch.nn.Module`
3. 在`src/utils/model_management.py`中注册新模型

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 贡献指南

欢迎提交Issue和Pull Request。在提交代码前,请确保:

1. 代码符合项目的代码风格(使用black和flake8)
2. 添加了适当的测试用例
3. 更新了相关文档

## 引用

如果您在研究中使用了本项目,请引用:

```bibtex
@software{HybridQCCNN2024,
  author = {BaizeXS},
  title = {HybridQCCNN: A Hybrid Quantum-Classical CNN Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BaizeXS/HybridQCCNN}
}
````
