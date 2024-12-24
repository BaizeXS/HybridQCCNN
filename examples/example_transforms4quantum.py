import torchvision

from utils import ToTensor4Quantum

if __name__ == "__main__":
    # Load Test Set
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        ToTensor4Quantum(),
    ])
    test_set = torchvision.datasets.FashionMNIST(root="../datasets", train=False, transform=transform, download=True)
    # Print Classes
    print(test_set.classes)
    # Select a Picture
    idx = 100
    image, label = test_set[idx]
    # print('img: ', image)
    print('label:', label)
    print('image.shape', image.shape)
