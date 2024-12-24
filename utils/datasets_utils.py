import json
import os
import random
import shutil


def extract_images_from_datasets(dataset, num_pics, out_dir='./pics/ext_pics/'):
    """
    Save a specified number of randomly selected images from a dataset to an output directory.
    """
    labels = dataset.classes
    random_indexes = random.sample(range(len(dataset)), num_pics)
    os.makedirs(out_dir, exist_ok=True)

    # Save data
    for idx in random_indexes:
        image, label = dataset[idx]
        class_name = labels[label]
        image_path = os.path.join(out_dir, class_name, f'{idx}.jpg')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        try:
            image.save(image_path)
        except Exception as e:
            print(f"Failed to save image {idx}: {str(e)}")


def save_class_indices(dataset, json_file_path):
    """
    Save the class information and their corresponding indices from the dataset to a JSON file.
    """
    labels = dataset.classes
    label_dict = {i: label for i, label in enumerate(labels)}

    with open(json_file_path, "w") as json_file:
        json.dump(label_dict, json_file, indent=4)

    print(f"Class labels and their indices successfully saved to {json_file_path}.")


def split_dataset(dataset_dir, output_dir, train_ratio=0.8):
    """将数据集划分为训练数据集和测试数据集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历数据集目录
    for category in os.listdir(dataset_dir):
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # 为每个类别创建子目录
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # 获取该类别下的所有图像文件
        files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        # 将图像文件分割为 train 和 test
        split_index = int(len(files) * train_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]

        # 将图像文件复制到对应的子目录
        for file in train_files:
            src = os.path.join(category_dir, file)
            dst = os.path.join(train_category_dir, file)
            shutil.copyfile(src, dst)

        for file in test_files:
            src = os.path.join(category_dir, file)
            dst = os.path.join(test_category_dir, file)
            shutil.copyfile(src, dst)

    print(f"Dataset split completed. Results saved in {output_dir}")
