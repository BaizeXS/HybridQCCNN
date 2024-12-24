from functools import wraps

import numpy as np
import torch
from torchvision.transforms import functional as TF


class ToTensor4Quantum:
    """
    Transform class to convert a PIL Image or ndarray to tensor and scale the values suitable for quantum computing.

    This class normalizes the input image and converts it to a tensor representation.
    The normalization process includes scaling pixel values in the image to the range (0, π).

    Usage:
        transform = ToTensor4Quantum()
        tensor_image = transform(image)
    """

    def __init__(self) -> None:
        pass

    @wraps(torch.tensor)
    def __call__(self, pic):
        return np.pi * TF.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
