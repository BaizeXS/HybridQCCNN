"""自定义数据集示例：从文件夹加载图像数据集。

目录结构示例：
data/
    train/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    test/
        class1/
            img5.jpg
        class2/
            img6.jpg
"""

from PIL import Image

from utils.data_management.dataset_manager import CustomDataset


class ImageFolderDataset(CustomDataset):
    """从文件夹加载图像数据集的示例实现"""

    def _load_data(self):
        """加载数据和标签

        遍历指定目录，将图像路径和对应的类别标签加载到内存中
        """
        # 确定数据子目录
        data_subdir = "train" if self.train else "test"
        data_path = self.data_dir / data_subdir

        # 获取所有类别
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 收集所有图像文件和对应的标签
        self.data = []  # 存储图像路径
        self.targets = []  # 存储标签

        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.data.append(img_path)
                    self.targets.append(class_idx)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        img_path = self.data[idx]
        target = self.targets[idx]

        # 加载图像
        img = Image.open(img_path).convert("RGB")

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, target
