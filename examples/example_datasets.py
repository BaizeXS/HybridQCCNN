import os

from utils import GarbageDataset

if __name__ == '__main__':
    # 测试数据集是否能够成功加载
    # Using PyCharm
    data_dir = "../datasets"
    # Using Command Line
    # data_dir = "datasets/GarbageDataset"
    train_set = GarbageDataset(root_dir=os.path.join(data_dir, 'GarbageDataset', 'train'))
    test_set = GarbageDataset(root_dir=os.path.join(data_dir, 'GarbageDataset', 'test'))
    print(len(train_set))
    print(len(test_set))
