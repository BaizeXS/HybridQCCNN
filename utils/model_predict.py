import random
import time

import torch
import torchvision
from torch.utils.data import Subset

from utils.model_utils import load_model_with_weights
from utils.transforms4quantum import ToTensor4Quantum
from utils.visualization import plot_probabilities

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型权重路径
models_weights = {
    'ClassicNet': '../output/ClassicNet/ClassicNet_model.pth',
    'ClassicNetUnscaled': '../output/ClassicNetUnscaled/ClassicNetUnscaled_model.pth',
    'HybridNet': '../output/HybridNet/HybridNet_model.pth',
    'HybridNetDeeper': '../output/HybridNetDeeper/HybridNetDeeper_model.pth',
    'HybridNetStrideOne': '../output/HybridNetStrideOne/HybridNetStrideOne_model.pth',
    'SimpleVGG': '../output/SimpleVGG/SimpleVGG_model.pth',
    'SimpleGoogLeNet': '../output/SimpleGoogLeNet/SimpleGooLeNet_model.pth',
    'HybridGoogLeNet': '../output/HybridGoogLeNet/HybridGoogLeNet_model.pth',
    'SimpleResNet': '../output/SimpleResNet/SimpleResNet_model.pth',
    'HybridResNet': '../output/HybridResNet/HybridResNet_model.pth',
}

# 初始化模型
models = {}
for model_name, weights_path in models_weights.items():
    model = load_model_with_weights(model_name, weights_path, device)
    if model is not None:
        models[model_name] = model

# 加载数据集
test_set = torchvision.datasets.FashionMNIST(root='../datasets', train=False, download=True)
class_info = test_set.classes

# 构建测试子集
test_subset_size = 64
indices = torch.randperm(len(test_set)).tolist()
subset_indices = indices[:test_subset_size]
test_subset = Subset(test_set, subset_indices)


def transform_image(image):
    """
    Transform the input image through a series of preprocessing steps and return it as a tensor.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((14, 14)),
        ToTensor4Quantum(),
    ])
    return transform(image).unsqueeze(0)  # Apply transformations and add a batch dimension


def get_prediction(image):
    """
    Predict the label of the given image and return the prediction results and probability distributions for each model.
    """
    # Disable gradient tracking for prediction
    with torch.no_grad():
        # Transform the input image to a tensor
        image_tensor = transform_image(image)

        # Initialize the dictionary to store results
        results = {}

        for _model_name, _model in models.items():
            start_time = time.time()  # 开始计时
            # Forward pass to get model outputs
            outputs = _model.forward(image_tensor)

            # Calculate Time
            elapsed_time = time.time() - start_time

            # Get the predicted label
            prediction = torch.argmax(outputs).item()
            prediction_label = class_info[prediction]

            # Calculate class probabilities and round to 4 decimal places
            probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy().round(4)

            # Store results in the dictionary
            results[_model_name] = {
                'prediction': prediction_label,
                'probabilities': probabilities.tolist(),
                'time': elapsed_time
            }

        return results


def main():
    # Randomly select 10 image indexes
    random_indexes = random.sample(range(len(test_set)), 10)
    print('Randomly generated 10 image indexes: ', random_indexes)

    # Get user input for image index
    index = int(input('Please enter the image index: '))
    image, label = test_set[index]

    # Prediction
    results = get_prediction(image)

    # Plot probability distribution chart
    plot_probabilities(results, class_info, save=False)

    # Output results
    print('Actual label: ', class_info[label])
    print('Predicted label: ')
    for _model_name in results.keys():
        print(f"\t{_model_name}: {results[_model_name]['prediction']}")


if __name__ == '__main__':
    main()
