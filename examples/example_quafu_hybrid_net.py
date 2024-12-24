import random
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from models.benchmark_hybrid import QuafuHybridNet
from utils import ToTensor4Quantum


def main():
    # 定义转换
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        ToTensor4Quantum(),
    ])

    # 加载测试集
    test_set = torchvision.datasets.FashionMNIST(root='../datasets', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # 定义类别标签
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    # 获取随机图片和标签
    index = random.randint(0, len(test_set) - 1)
    image, label = test_set[index]

    # 显示图片及其标签
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label: {classes[label]}")
    plt.show()

    # 载入模型
    use_quantum_cloud = True  # 选择是否使用夸父云平台
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    api_token = ("ccE4eawT5dsWtM359uqnDQX6vBVvVDMgIrPlKwAmT2x.Qf0cjMzgTO4EzNxojIwhXZiwCMzgTM6ICZpJye"
                 ".9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
    qdevice = "Dongling" if use_quantum_cloud else "simulator"
    model_weights_path = "../output/HybridNetDeeper/HybridNetDeeper_model.pth"
    model = QuafuHybridNet(num_classes=10, qdevice=qdevice, api_token=api_token)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 执行前向传播以获得预测
    with torch.no_grad():
        image = image.unsqueeze(0)  # 添加批次维度

        start_time = time.time()
        outputs = model(image)
        end_time = time.time()
        execution_time = end_time - start_time

        _, predicted = torch.max(outputs, 1)
        predicted_label = classes[predicted.item()]

    print(f"True Label: {classes[label]}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Execute time: {execution_time: .2f}s.")


if __name__ == '__main__':
    main()
