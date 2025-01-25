# HybridQCCNN

HybridQCCNN is a hybrid quantum-classical convolutional neural network framework designed for image classification. It integrates multiple predefined models, supports various quantum simulator backends, and provides rich visualization and analysis tools to facilitate both classical and quantum-enhanced training workflows.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Common Commands](#common-commands)
- [Developer Guide](#developer-guide)
- [Project Structure](#project-structure)
- [Configuration Files](#configuration-files)
- [Customization](#customization)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

1. Unified training and testing for both classical CNNs and hybrid quantum-classical CNNs
2. Multiple predefined models (e.g., VGG, GoogLeNet, ResNet) in both classic and quantum-hybrid forms
3. Support for custom datasets and architectures
4. Complete training visualization and model performance comparison
5. Multiple quantum simulators supported (e.g., PennyLane, Qiskit)

## Installation

### Requirements

- Python >= 3.10
- (Optional) CUDA >= 12.1 for GPU acceleration
- OS: Linux (recommended), Windows, macOS

### Steps

1. **Clone Repository**

   ```bash
   git clone https://github.com/BaizeXS/HybridQCCNN.git
   cd HybridQCCNN
   ```

2. **Create and Activate Environment**

   ```bash
   conda create -n qml python=3.10
   conda activate qml
   ```

3. **Install Dependencies**

   ```bash
   pip install -e ".[all-dev]"
   ```

4. **Verify Installation**

   ```bash
   python -m src.main --help
   ```

### Key Dependencies

- pennylane ~= 0.40.0
- torch ~= 2.5.1
- torchvision ~= 0.20.1
- numpy ~= 2.0.2
- pandas ~= 2.2.3
- scikit-learn ~= 1.6.0
- matplotlib ~= 3.10.0
- tensorboard >= 2.18.0
- qutip ~= 5.1.1

## Quick Start

Here is a quick demo on how to train and test a hybrid quantum-classical model using `configs/benchmark/hybrid_fashionmnist.yaml`:

1. **Edit Configuration File**
   Adjust parameters such as batch size and learning rate as needed.

2. **Train Model**
   ```bash
   python -m src.main train -c configs/benchmark/hybrid_fashionmnist.yaml
   ```

3. **Test Model**

   ```bash
   python -m src.main test -c configs/benchmark/hybrid_fashionmnist.yaml \
       -w outputs/HybridNet/weights/best_model.pt
   ```

4. **Predict on a Single Image**
   ```bash
   python -m src.main predict -c configs/benchmark/hybrid_fashionmnist.yaml \
       -w outputs/hybrid_fashionmnist/weights.pth -i path/to/image.jpg
   ```

## Common Commands

### Training

```bash
python -m src.main train -c <CONFIG> [--checkpoint <CHECKPOINT_PATH>]
```

### Testing

```bash
python -m src.main test -c <CONFIG> -w <WEIGHTS_PATH> [--is-checkpoint]
```

### Single Image Prediction

```bash
python -m src.main predict -c <CONFIG> -w <WEIGHTS_PATH> -i <IMAGE_PATH> [--is-checkpoint]
```

### Visualization of Training Metrics

```bash
python -m src.main viz-metrics -f <METRICS_FILE> [--metric-names <NAMES>] \
    [--phases <PHASES>] [-o <OUTPUT_DIR>] [--no-show]
```

### Model Performance Comparison

```bash
python -m src.main compare -f <METRICS_FILES>... [--metric-names <NAMES>] \
    [--model-names <NAMES>] [--phases <PHASES>] [-o <OUTPUT_DIR>] [--no-show]
```

## Developer Guide

Install developer tools and set up pre-commit hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- **black**: Code formatter (88-line width)
- **isort**: Sort import statements
- **flake8**: Code style checks
- **pre-commit**: Automates the above checks

Run tests:

```bash
pytest          # Run all tests
pytest -m unit  # Run only unit tests
```

Supported markers include `slow`, `gpu`, `quantum`, `integration`, `unit`, and `heavy_model`.

## Project Structure

```
HybridQCCNN/
├── src/                # Source code (models, trainers, utilities)
├── tests/              # Test files
├── configs/            # YAML configuration files
├── examples/           # Example scripts
├── docs/               # Documentation
└── datasets/           # Dataset folder
```

## Configuration Files

Typical YAML configuration includes:

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
  model_type: "classic"  # or "hybrid"
  quantum_config:
    q_layers: 2
    diff_method: "backprop"
    q_device: "default.qubit"

training:
  learning_rate: 0.001
  weight_decay: 1e-4
  num_epochs: 20

device: "cpu"  # or "cuda", "mps"
seed: 42
output_dir: "outputs/experiment_name"
```

Create separate YAML files under `configs/` for different models or datasets.

## Customization

### Add a New Dataset

To add a new dataset:

1. Create a dataset class file (e.g., `src/datasets/my_dataset.py`) that inherits from `torch.utils.data.Dataset`:

   ```python
   from utils.data_management import CustomDataset

   class MyDataset(CustomDataset):
       def _load_data(self):
           # Implement data loading logic
           pass

       def __getitem__(self, index):
           # Implement data retrieval logic
           pass
   ```

2. In the configuration file, set the dataset parameters:

   ```yaml
   data:
     name: "MyDataset"  # Must match the dataset class name
     dataset_type: "CUSTOM"  # Must be set to CUSTOM
     input_shape: [3, 224, 224]  # Adjust based on actual input dimensions
     num_classes: 10
     dataset_path: "path/to/dataset"  # Root directory of your dataset
     custom_dataset_path: "path/to/my_dataset.py"  # File path of the dataset class
     train_split: 0.8
     batch_size: 32
     # Other data configurations...
   ```

### Add a New Model

To add a new model:

1. Create a model class file (e.g., `src/models/my_model.py`) that inherits from `torch.nn.Module`:

   ```python
   import torch.nn as nn

   class MyModel(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           # Implement your model architecture

       def forward(self, x):
           # Implement the forward pass
           return x
   ```

2. In the configuration file, set the model parameters:

   ```yaml
   model:
     name: "MyModel"  # Must match the model class name
     model_type: "custom"  # Must be set to custom
     model_kwargs:  # Model initialization parameters
       num_classes: 10
       hidden_dim: 128
     custom_model_path: "path/to/my_model.py"  # File path of the model class
     quantum_config: null  # Needed only if it's a quantum hybrid model
   ```

For a complete configuration file example, see `templates/custom_config.yaml`.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it for both commercial and non-commercial purposes.

## Citation

If this project helps your research, please cite:

```bibtex
@software{HybridQCCNN2024,
  author = {BaizeXS},
  title = {HybridQCCNN: A Hybrid Quantum-Classical CNN Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BaizeXS/HybridQCCNN}
}
```
