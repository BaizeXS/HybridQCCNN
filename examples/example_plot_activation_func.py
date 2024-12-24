import numpy as np

from utils.visualization import plot_activation_function


# ReLU 函数
def relu(x):
    return np.maximum(0, x)


# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # plot_activation_function(relu, 'ReLU')
    plot_activation_function(sigmoid, 'Sigmoid')
