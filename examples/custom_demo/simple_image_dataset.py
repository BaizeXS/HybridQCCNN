from PIL import Image

from utils.data_management import CustomDataset


class SimpleImageDataset(CustomDataset):
    def _load_data(self):
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
        img_path = self.data[index]
        target = self.targets[index]

        # Load image from file
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
