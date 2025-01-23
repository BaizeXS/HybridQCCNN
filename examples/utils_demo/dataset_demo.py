from pathlib import Path

from PIL import Image

from config import DataConfig
from utils.data_management import CustomDataset, DatasetManager


def basic_dataset_example():
    """
    Basic example using built-in MNIST dataset
    """
    config = DataConfig(
        name="MNIST",
        input_shape=(1, 28, 28),
        num_classes=10,
        dataset_type="MNIST",
        batch_size=32,
        train_split=0.8,
        num_workers=2,
        pin_memory=True,
        train_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
        val_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
        test_transforms=[
            {"name": "ToTensor"},
            {"name": "Normalize", "args": {"mean": [0.1307], "std": [0.3081]}},
        ],
    )

    manager = DatasetManager(config, data_dir="./datasets")  # type: ignore
    train_loader, val_loader, test_loader = manager.get_data_loaders()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


class SimpleImageDataset(CustomDataset):
    """Example implementation of CustomDataset"""

    def _load_data(self):
        """Load image data from directory structure"""
        data_subdir = "train" if self.train else "test"
        data_path = self.data_dir / data_subdir

        # Get class names from directory names
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect image files and labels
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_idx = class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.data.append(img_path)
                    self.targets.append(class_idx)

    def __getitem__(self, index):
        """Get a single item from the dataset.

        Args:
            index (int): Index of the item to get

        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path = self.data[index]
        target = self.targets[index]

        # Load image from file
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def custom_dataset_example():
    """Example of using custom dataset implementation"""
    # Create custom dataset file
    custom_dataset_dir = Path("./templates")
    custom_dataset_dir.mkdir(exist_ok=True)

    # Create data directories
    data_root = Path("./datasets/images")
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    # Create class directories and sample images
    classes = ["class1", "class2"]
    for phase_dir in [train_dir, test_dir]:
        for class_name in classes:
            class_dir = phase_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Create a dummy image for demonstration
            img = Image.new("RGB", (224, 224), color="white")
            for i in range(3):  # Create 3 sample images per class
                img.save(class_dir / f"sample_{i}.jpg")

    # Create custom dataset implementation
    dataset_file = custom_dataset_dir / "simple_image_dataset.py"
    dataset_code = """from utils.data_management import CustomDataset
from PIL import Image

class SimpleImageDataset(CustomDataset):
    def _load_data(self):
        data_subdir = 'train' if self.train else 'test'
        data_path = self.data_dir / data_subdir

        # Get class names from directory names
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect image files and labels
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_idx = class_to_idx[class_dir.name]
                for img_path in class_dir.glob('*.jpg'):
                    self.data.append(img_path)
                    self.targets.append(class_idx)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]

        # Load image from file
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
"""
    dataset_file.write_text(dataset_code)

    config = DataConfig(
        name="SimpleImageDataset",  # Must be the same as the dataset class name
        input_shape=(3, 224, 224),
        num_classes=2,  # Match the number of classes we created
        dataset_type="CUSTOM",
        dataset_path=data_root,  # Data directory path
        custom_dataset_path=dataset_file,  # Custom dataset class file path
        batch_size=16,
        train_split=0.9,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        train_transforms=[
            {"name": "Resize", "args": {"size": (224, 224)}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        ],
        val_transforms=[
            {"name": "Resize", "args": {"size": (224, 224)}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        ],
        test_transforms=[
            {"name": "Resize", "args": {"size": (224, 224)}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        ],
    )

    manager = DatasetManager(config, data_dir=data_root)
    train_loader, val_loader, test_loader = manager.get_data_loaders()

    # Display dataset information
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")


def transformation_example():
    """
    Example of configuring different data transformations
    """
    # Define different transforms for training and validation
    train_transforms = [
        {"name": "RandomResizedCrop", "args": {"size": 224}},
        {"name": "RandomHorizontalFlip"},
        {"name": "ToTensor"},
        {
            "name": "Normalize",
            "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
    ]

    val_transforms = [
        {"name": "Resize", "args": {"size": 256}},
        {"name": "CenterCrop", "args": {"size": 224}},
        {"name": "ToTensor"},
        {
            "name": "Normalize",
            "args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
    ]

    config = DataConfig(
        name="CIFAR10",
        input_shape=(3, 32, 32),  # CIFAR10 image size
        num_classes=10,
        dataset_type="CIFAR10",
        batch_size=32,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,  # Use same transforms as validation for test
    )

    manager = DatasetManager(config, data_dir="./datasets")  # type: ignore
    train_loader, val_loader, test_loader = manager.get_data_loaders()


if __name__ == "__main__":
    print("Basic dataset example:")
    basic_dataset_example()

    print("\nCustom dataset example:")
    custom_dataset_example()

    print("\nTransformation configuration example:")
    transformation_example()
