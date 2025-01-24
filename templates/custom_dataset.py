"""Custom dataset example: Load image dataset from a folder.

Example of directory structure:
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
    """Example implementation of loading image dataset from a folder"""

    def _load_data(self):
        """Load data and labels

        Iterate over the specified directory, load image paths and corresponding
        class labels into memory
        """
        # Determine data subdirectory
        data_subdir = "train" if self.train else "test"
        data_path = self.data_dir / data_subdir

        # Get all classes
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all image files and corresponding labels
        self.data = []  # Store image paths
        self.targets = []  # Store labels

        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.data.append(img_path)
                    self.targets.append(class_idx)

    def __getitem__(self, idx):
        """Get a single data sample"""
        img_path = self.data[idx]
        target = self.targets[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, target
