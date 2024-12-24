from utils.datasets_utils import split_dataset

if __name__ == "__main__":
    dataset_dir = "../datasets/TestDataset"
    output_dir = "../datasets/TestDatasetSplit"
    train_ratio = 0.8

    split_dataset(dataset_dir, output_dir, train_ratio)
