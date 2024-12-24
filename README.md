# HybridQCCNN

## Introduction



## Environment Configuration

...

### 1. Set up Conda Environment

Followed the [Quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) on the official website.

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Initialize for bash and zsh shells

```shell
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

==Then exit and create a new session.==

Create a virtual environment

```shell
conda create -n qml python=3.10
```

Activate the environment

```shell 
conda activate qml
```



### 2. Install Dependencies

##### Work with CPU

```shell
# PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Basic ML Tools
conda install -c conda-forge numpy matplotlib pandas scipy scikit-learn tqdm tensorboard

# Pennylane
pip install pennylane --upgrade
# Pennylane CPU Plugins
pip install pennylane-lightning
pip install pennylane-lightning[kokkos]
# pip install pennylane-qulacs["cpu"]

# Quafu
pip install pyquafu
# Seaborn

```

##### Work with GPU

```shell
# PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Basic ML Tools
conda install -c conda-forge numpy matplotlib pandas scipy scikit-learn tqdm pybind11

# Pennylane
pip install pennylane --upgrade

# Pennylane CPU Plugins
pip install pennylane-lightning
pip install pennylane-lightning[kokkos]
# pip install pennylane-qulacs["cpu"]

# cuQuantum
pip install -v --no-cache-dir cuquantum-python-cu12
pip install nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cudnn-cu12==8.9.2.26 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-nccl-cu12==2.19.3 nvidia-nvtx-cu12==12.1.105 nvidia-cusparse-cu12==12.1.0.106 nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-runtime-cu12==12.1.105 custatevec_cu12
# pip install nvidia-cusparse-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 custatevec_cu12

# Pennylane GPU Plugins
pip install pennylane-lightning[gpu]
# pip install pennylane-qulacs["gpu"]
```





## Train the model

### Train the model through a daemon using `screen`

##### Installation

```shell
sudo apt update && sudo apt install -y screen
```

##### Usage

Create a new session.

```shell
screen -S train
```

Detache from the current session.

```shell
[Ctrl + A + D]
```

List all the sessions.

```shell
screen -ls
```

Reattach to a detached session.

```shell
screen -r train
```



### Parallel training model with OpenMP

##### Installation

```shell
sudo apt -y update && sudo apt install g++ libomp-dev
```

##### Set up OpenMP

```shell
# TODO: Check out which are valid
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```



### Train the model

```shell
# Available Quantum Simulators:
# default.qubit, default.qubit.torch, lightning.qubit, lightning.gpu, lightning.kokkos
# Available Quantum Differential Method:
# best, backprop, adjoint, parameter-shift

# Example 1
python model_train.py --model HybridNet --qdevice lightning.qubit --qdevice-kwargs '{"batch_obs": true}' --diff-method adjoint
# Example 2
python model_train.py --model HybridNet --qdevice lightning.qubit --qdevice-kwargs '{"shots": 1000, "mcmc": true}' --diff-method adjoint
# Example 3
python model_train.py --model HybridNet --qdevice lightning.qubit --qdevice-kwargs '{"batch_obs": true, "shots": 1000, "mcmc": true, "kernel_name": "NonZeroRandom", "num_burnin": 200}' --diff-method adjoint
```



```shell
如果将inception4a也改成qubit -> 39s
如果将inception5a也改成qubit -> 42s
说明只要用多个小的卷积核就可以 -> 考虑并行计算
但是这样的话至少30GB内存


一种构想：
现阶段可用的卷积核
-   3 channel 2*2 => 12
-   4 channel 2*2 => 16
-   2 channel 3*3 => 18

1. 构建ResNext: 192核心 256GB 尝试并行多个 小规模 Quanv2d
2. 先降为在升维，类似于先编码再解码的工作，充分利用Quanv2d
```



```python
# HybridNet
# Using default.qubit
python -m utils.model_train --model HybridNet --dataset FashionMNIST --data-dir datasets --epochs 10 --batch-size 64 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice default.qubit --diff-method best
# Using default.qubit.torch
python -m utils.model_train --model HybridNet --dataset FashionMNIST --data-dir datasets --epochs 10 --batch-size 64 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice default.qubit.torch --diff-method best
# Using lightning.qubit
python -m utils.model_train --model HybridNet --dataset FashionMNIST --data-dir datasets --epochs 10 --batch-size 64 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice lightning.qubit --diff-method adjoint
# Using lightning.kokkos
python -m utils.model_train --model HybridNet --dataset FashionMNIST --data-dir datasets --epochs 10 --batch-size 64 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice lightning.kokkos --diff-method adjoint
# Using qulacs.simulator
python -m utils.model_train --model HybridNet --dataset FashionMNIST --data-dir datasets --epochs 10 --batch-size 64 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice qulacs.simulator --diff-method best

# SimpleVGG
python -m utils.model_train --model SimpleVGG --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs
# SimpleGoogLeNet
python -m utils.model_train --model SimpleGoogLeNet --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs
# SimpleResNet
python -m utils.model_train --model SimpleResNet --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs
# HybridVGG
python -m utils.model_train --model HybridVGG --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice lightning.kokkos --diff-method adjoint
# HybridGoogLeNet
python -m utils.model_train --model HybridGoogLeNet --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice lightning.kokkos --diff-method adjoint
# HybridResNet
python -m utils.model_train --model HybridResNet --dataset GarbageDataset --data-dir datasets --epochs 15 --batch-size 32 --seed 42 --output-dir output --tensorboard-dir logs --device cpu --qdevice lightning.kokkos --diff-method adjoint








# Test Benchmark
# ClassicNet
python -m utils.model_test --model ClassicNet --dataset FashionMNIST --data-dir datasets --batch-size 64 --seed 42 --output-dir output --device cpu
# HybridNet
python -m utils.model_test --model HybridNet --dataset FashionMNIST --data-dir datasets --batch-size 64 --seed 42 --output-dir output --device cpu --qdevice lightning.kokkos --diff-method adjoint
# HybridNetDeeper
python -m utils.model_test --model HybridNetDeeper --dataset FashionMNIST --data-dir datasets --batch-size 64 --seed 42 --output-dir output --device cpu --qdevice lightning.kokkos --diff-method adjoint

# Test Garbage Classification
# SimpleVGG
python -m utils.model_test --model SimpleVGG --dataset GarbageDataset --data-dir datasets --batch-size 32 --seed 42 --output-dir output --device cpu
# SimpleGoogLeNet
python -m utils.model_test --model SimpleGoogLeNet --dataset GarbageDataset --data-dir datasets --batch-size 32 --seed 42 --output-dir output --device cpu
# SimpleResNet
python -m utils.model_test --model SimpleResNet --dataset GarbageDataset --data-dir datasets --batch-size 32 --seed 42 --output-dir output --device cpu
# HybridGoogLeNet
python -m utils.model_test --model HybridGoogLeNet --dataset GarbageDataset --data-dir datasets --batch-size 32 --seed 42 --output-dir output --device cpu --qdevice lightning.kokkos --diff-method adjoint
# HybridResNet
python -m utils.model_test --model HybridResNet --dataset GarbageDataset --data-dir datasets --batch-size 32 --seed 42 --output-dir output --device cpu --qdevice lightning.kokkos --diff-method adjoint
```





