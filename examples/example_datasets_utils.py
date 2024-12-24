import torchvision
from torchvision.datasets import FashionMNIST

from utils import GarbageDataset, extract_images_from_datasets, save_class_indices

if __name__ == '__main__':
    # 加载测试数据集
    test_fmnist_dataset: FashionMNIST = torchvision.datasets.FashionMNIST(root='../datasets', train=False, download=True)
    test_garbage_dataset = GarbageDataset(root_dir="../datasets/GarbageDataset/test")
    test_dataset = test_fmnist_dataset
    # 从数据集中提取一定数量的图像并将它们按照类别保存到指定目录
    extract_images_from_datasets(test_fmnist_dataset, 50, out_dir='../pics/test_pics/fmnist/')
    extract_images_from_datasets(test_garbage_dataset, 50, out_dir='../pics/test_pics/garbage/')
    # 保存数据集中的类别信息和索引
    save_class_indices(test_garbage_dataset, "../backend/FashionMNIST_Class_Info.json")
    save_class_indices(test_garbage_dataset, "../backend/GarbageDataset_Class_Info.json")
