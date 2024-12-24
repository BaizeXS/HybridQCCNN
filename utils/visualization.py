import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
from qutip import Bloch

from utils.model_utils import load_evaluation_metrics

# Constants
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
FIGSIZE_WIDTH = 12
FIGSIZE_HEIGHT_PER_ROW = 3
PROBABILITY_FIGSIZE_WIDTH = 18
PROBABILITY_FIGSIZE_HEIGHT_PER_ROW = 6


def set_plot_style(ax, title, xlabel, ylabel):
    """
    Set the style of a subplot.
    """
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)


def plot_evaluation_metrics(models_linestyles, data_types, names=None, data_dir='./output/', show=True, save=False,
                            save_path='./output/evaluation_metrics_chart.png'):
    """
    Plot models' evaluation metrics as line charts.
    """
    # Set subplot layout
    num_data_types = len(data_types)
    num_columns = 2
    num_rows = (num_data_types + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT_PER_ROW * num_rows))
    axes = axes.flatten()

    # Plot subplots
    for idx, data_type in enumerate(data_types):
        ax = axes[idx]
        title = data_type.replace('_', ' ').title()
        ylabel = data_type.split('_')[1].title()
        set_plot_style(ax, title, 'Epoch', ylabel)

        for model_name, linestyle in models_linestyles.items():
            data = load_evaluation_metrics(model_name, data_type, data_dir)
            # ax.plot(data, label=model_name, linestyle=linestyle)
            label = names.get(model_name, model_name) if names else model_name
            ax.plot(data, label=label, linestyle=linestyle)
            ax.legend(fontsize=LEGEND_FONTSIZE)

    for i in range(num_data_types, num_columns * num_rows):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()


def save_confusion_matrix(confusion_matrix_data, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probabilities(results, labels, show=True, save=False, save_path='./probabilities.png', colors=None):
    """
    Plot probability distribution charts for multiple model classification results.
    """
    if not results:
        print("No results to plot.")
        return

    # Default color list
    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    # Set subplot layout
    num_models = len(results)
    num_columns = 3
    num_rows = (num_models + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns,
                             figsize=(PROBABILITY_FIGSIZE_WIDTH, PROBABILITY_FIGSIZE_HEIGHT_PER_ROW * num_rows))
    axes = axes.flatten()

    # Plot probability distribution charts
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]  # 获取当前子图

        probabilities = result['probabilities'][0]
        ax.bar(labels, probabilities, color=colors[idx % len(colors)])
        set_plot_style(ax, f'{model_name} Prediction', '', '')
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 0.3)

    # Clear unused subplots
    for i in range(num_models, num_columns * num_rows):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save probability distribution charts
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    # Show probability distribution charts
    if show:
        plt.show()


def plot_activation_function(activation_func, func_name):
    # 生成输入数据
    x = np.linspace(-10, 10, 400)

    # 计算激活函数的值
    y = activation_func(x)
    y_range = max(abs(min(y)), abs(max(y))) + 0.1

    # 创建图形
    plt.figure(figsize=(6, 4))

    # 随机选择颜色
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    color = random.choice(colors)

    # 绘制激活函数图像
    plt.plot(x, y, label=func_name, color=color)
    plt.title(f'{func_name} Function')
    plt.xlabel('x')
    plt.ylabel(f'{func_name}(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-10, 10)
    # plt.ylim(min(y) - 0.1, max(y) + 0.1)
    plt.ylim(-y_range, y_range)
    plt.legend()
    plt.grid()

    # 显示图形
    plt.tight_layout()
    plt.show()


def plot_bloch_sphere(state=None):
    # 创建布洛赫球对象
    bloch_sphere = Bloch()

    # 自定义布洛赫球的外观
    bloch_sphere.frame_color = 'gray'  # 改变球架的颜色
    bloch_sphere.font_size = 16
    bloch_sphere.sphere_color = 'lightblue'  # 改变球体的颜色
    bloch_sphere.vector_color = ['green']  # 定义向量的颜色
    bloch_sphere.vector_width = 2  # 向量的线条宽度
    bloch_sphere.figsize = [9, 9]
    # 添加三个基本轴向量来突出显示
    bloch_sphere.add_vectors([1, 0, 0])  # X轴
    bloch_sphere.add_vectors([0, 1, 0])  # Y轴
    bloch_sphere.add_vectors([0, 0, 1])  # Z轴

    # 如果提供了状态向量，则添加到布洛赫球中
    if state is None:
        bloch_sphere.add_vectors(state)

    # 显示布洛赫球
    bloch_sphere.show()


def plot_qcircuit(qnode, inputs, weights, style="pennylane"):
    qml.drawer.use_style(style)
    fig, ax = qml.draw_mpl(qnode)(inputs, weights)
    plt.show()
