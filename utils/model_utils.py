import logging
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from models import *
from models import ALL_MODELS
from utils.transforms4quantum import ToTensor4Quantum


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_models(dataset_name, qdevice, qdevice_kwargs, diff_method):
    if dataset_name == 'FashionMNIST':
        return {
            'ClassicNet': ClassicNet(num_classes=10),
            'HybridNet': HybridNet(num_classes=10, qkernel=None, num_qlayers=1,
                                   qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
            'HybridNetDeeper': HybridNet(num_classes=10, qkernel=None, num_qlayers=2,
                                         qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
            'HybridNetStrideOne': HybridNet(num_classes=10, qkernel=None, num_qlayers=2, stride=1,
                                            qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
            # TODO: About Barren Plateau
            'HybridNetDeeper2': HybridNet(num_classes=10, qkernel=None, num_qlayers=3,
                                          qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
        }
    elif dataset_name == 'GarbageDataset':
        return {
            'SimpleVGG': SimpleVGG(num_classes=10),
            # 'HybridVGG': HybridVGG(num_classes=10, qkernel=None, num_qlayers=2, qdevice=qdevice,
            #                        qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
            'SimpleGoogLeNet': SimpleGoogLeNet(num_classes=10),
            'HybridGoogLeNet': HybridGoogLeNet(num_classes=10, qkernel=None, num_qlayers=2,
                                               qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
            'SimpleResNet': simple_resnet18(num_classes=10),
            'HybridResNet': hybrid_resnet18(num_classes=10, qkernel=None, num_qlayers=2,
                                            qdevice=qdevice, qdevice_kwargs=qdevice_kwargs, diff_method=diff_method),
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_transforms(model_name, dataset_name):
    model_type = 'Hybrid' if 'Hybrid' in model_name else 'Classic'
    if dataset_name == 'FashionMNIST' and model_type == 'Classic':
        train_transform = test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((14, 14)),
            torchvision.transforms.ToTensor(),
        ])
    elif dataset_name == 'FashionMNIST' and model_type == 'Hybrid':
        train_transform = test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((14, 14)),
            ToTensor4Quantum(),
        ])
    elif dataset_name == 'GarbageDataset' and model_type == 'Classic':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(72),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=15),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(72),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
        ])
    elif dataset_name == 'GarbageDataset' and model_type == 'Hybrid':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(72),
            torchvision.transforms.RandomCrop(64),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=15),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToTensor4Quantum(),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(72),
            torchvision.transforms.CenterCrop(64),
            ToTensor4Quantum(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_transform, test_transform


def get_dataset(dataset_name, data_dir, train_transform, test_transform, batch_size):
    if dataset_name == 'FashionMNIST':
        train_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=train_transform,
                                                      download=True)
        test_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=test_transform,
                                                     download=True)
    elif dataset_name == 'GarbageDataset':
        train_set = GarbageDataset(root_dir=os.path.join(data_dir, 'GarbageDataset', 'train'),
                                   transform=train_transform)
        test_set = GarbageDataset(root_dir=os.path.join(data_dir, 'GarbageDataset', 'test'),
                                  transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def construct_file_path(model_name, data_type, data_dir='./output/'):
    """
    Construct the file path based on model name and data type.
    """
    return os.path.join(data_dir, model_name, f'{model_name}_{data_type}.npy')


def load_model_with_weights(model_name, model_weights_path, num_classes, device, **kwargs):
    """
    Load weights data into the specified model and return the model.
    """
    try:
        if model_name not in ALL_MODELS:
            raise ValueError("无效的模型名称")

        model = ALL_MODELS[model_name](num_classes=num_classes, **kwargs)

        try:
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
        except FileNotFoundError:
            logging.error(f"Model weights file not found: {model_weights_path}")
            return None

        model.to(device)
        model.eval()

        return model

    except Exception as e:
        logging.error(f"Error loading the model: {str(e)}")
        return None


def save_evaluation_metrics(model_name, data, data_type, data_dir='./output/'):
    """
    Save evaluation metrics to the specified path.
    """
    data_file_path = construct_file_path(model_name, data_type, data_dir)
    os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
    np.save(data_file_path, data)


def load_evaluation_metrics(model_name, data_type, data_dir='./output/'):
    """
    Load evaluation metrics data from a file.
    """
    data_file_path = construct_file_path(model_name, data_type, data_dir)
    if os.path.exists(data_file_path):
        return np.load(data_file_path)
    else:
        logging.warning(f"Evaluation metrics file not found: {data_file_path}")
        return None
