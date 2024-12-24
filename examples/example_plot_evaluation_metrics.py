from utils import plot_evaluation_metrics


def plot_figure_1():
    # Configuration
    models_linestyles = {
        'ClassicNet': '-',
        'HybridNet': '-.',
        'HybridNetDeeper': ':',
        'HybridNetStrideOne': '--',
    }
    names = {
        "ClassicNet": 'CNN',
        'HybridNet': 'HQCCNN-1',
        'HybridNetDeeper': 'HQCCNN-2',
        'HybridNetStrideOne': 'HQCCNN-3',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types, names=names,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


def plot_figure_2():
    # Configuration
    models_linestyles = {
        'SimpleGoogLeNet': '-',
        'SimpleResNet': '-.',
        'HybridGoogLeNet': ':',
        'HybridResNet': '--',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


def plot_figure_3():
    # Configuration
    models_linestyles = {
        'SimpleVGG': '-',
        'SimpleVGG20': '-.',
        'SimpleVGG30': ':',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


def plot_figure_4():
    # Configuration
    models_linestyles = {
        'SimpleGoogLeNet': '-',
        'SimpleGoogLeNet20': '-.',
        'SimpleGoogLeNet30': ':',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


def plot_figure_5():
    # Configuration
    models_linestyles = {
        'SimpleResNet': '-',
        'SimpleResNet20': '-.',
        'SimpleResNet30': ':',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


def plot_figure_6():
    # Configuration
    models_linestyles = {
        'SimpleGoogLeNet': '-',
        # 'SimpleResNet': '-.',
        'HybridGoogLeNet': ':',
        # 'HybridResNet': '--',
    }
    data_types = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss']
    # Evaluation metrics visualization
    plot_evaluation_metrics(models_linestyles=models_linestyles, data_types=data_types,
                            data_dir='../output', save=False, save_path='./evaluation_metrics_chart.png')


if __name__ == '__main__':
    # plot_figure_1()
    # plot_figure_2()
    # plot_figure_3()
    # plot_figure_4()
    # plot_figure_5()
    plot_figure_6()
