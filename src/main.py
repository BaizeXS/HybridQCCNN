"""主程序入口点。

此模块提供:
1. 配置加载和解析
2. 数据集准备
3. 模型训练和评估的主流程
4. 结果可视化
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import ConfigManager
from utils.model_management import ModelManager
from utils.visualization import MetricsPlotter

def parse_args():
    """解析命令行参数。
    
    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='量子-经典混合神经网络训练程序')
    
    # 基础配置
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model_name', type=str, default='default', help='模型实例名称')
    
    # 数据集配置
    parser.add_argument('--data_dir', type=str, default='datasets', help='数据集根目录')
    parser.add_argument('--dataset', type=str, help='数据集类型(覆盖配置文件)')
    parser.add_argument('--batch_size', type=int, help='批次大小(覆盖配置文件)')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, help='训练轮数(覆盖配置文件)')
    parser.add_argument('--device', type=str, help='运行设备(覆盖配置文件)')
    parser.add_argument('--seed', type=int, help='随机种子(覆盖配置文件)')
    parser.add_argument('--num_workers', type=int, help='数据加载线程数(覆盖配置文件)')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    parser.add_argument('--local_rank', type=int, default=0, help='本地GPU序号')
    parser.add_argument('--world_size', type=int, help='总进程数')
    
    # 检查点配置
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--evaluate', action='store_true', help='仅进行评估')
    
    # 其他选项
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--amp', action='store_true', help='使用自动混合精度训练')
    
    return parser.parse_args()

def setup_logging():
    """配置日志记录。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_dataset(config, data_dir, transform):
    """根据配置获取数据集。
    
    Args:
        config: DataConfig对象
        data_dir: 数据集目录
        transform: 数据转换
        
    Returns:
        Dataset对象
    """
    if config.dataset_type.upper() == 'CIFAR10':
        return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif config.dataset_type.upper() == 'MNIST':
        return datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif config.dataset_type.upper() == 'CUSTOM':
        if not config.dataset_path:
            raise ValueError("自定义数据集必须提供dataset_path")
        # 实现自定义数据集加载逻辑
        raise NotImplementedError("自定义数据集加载尚未实现")
    else:
        raise ValueError(f"未知的数据集类型: {config.dataset_type}")

def get_transforms(transform_configs):
    """Convert transform configurations to actual transforms.
    
    Args:
        transform_configs: List of transform configurations
        
    Returns:
        transforms.Compose object
    """
    transform_list = []
    for t_config in transform_configs:
        transform_name = t_config['name']
        transform_args = t_config.get('args', {})
        
        if hasattr(transforms, transform_name):
            transform = getattr(transforms, transform_name)(**transform_args)
            transform_list.append(transform)
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
            
    return transforms.Compose(transform_list)

def prepare_data(config, data_dir):
    """准备数据集和数据加载器。"""
    # 获取数据转换
    train_transform = get_transforms(config.data.train_transforms)
    val_transform = get_transforms(config.data.val_transforms)
    test_transform = get_transforms(config.data.test_transforms)
    
    # 加载训练数据集
    train_dataset = get_dataset(config.data, data_dir, train_transform)
    
    # 加载测试数据集
    test_dataset = get_dataset(
        config.data, 
        data_dir, 
        test_transform,
        train=False
    )
    
    # 使用新增的train_split字段分割训练集和验证集
    train_size = int(config.data.train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # 使用随机种子确保可重复性
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # 设置验证集的转换
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器，使用新增的num_workers和pin_memory字段
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader, test_loader

def main():
    """主程序入口。"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger('main')
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 更新配置(如果命令行参数提供)
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.dataset:
        config.data.dataset_type = args.dataset
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    
    # 设置设备
    if args.no_cuda:
        config.device = 'cpu'
    elif torch.cuda.is_available():
        if args.distributed:
            config.device = f'cuda:{args.local_rank}'
        else:
            config.device = 'cuda'
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 准备数据
    data_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = prepare_data(config, data_dir)
    
    # 创建模型管理器
    model_manager = ModelManager(config, args.model_name)
    
    # 如果需要分布式训练
    if args.distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.local_rank
        )
        model_manager.model = torch.nn.parallel.DistributedDataParallel(
            model_manager.model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    
    # 如果需要恢复训练
    if args.resume:
        epoch = model_manager.load_checkpoint(args.resume)
        logger.info(f"从轮次 {epoch} 恢复训练")
    
    # 如果仅进行评估
    if args.evaluate:
        logger.info("开始评估...")
        test_metrics = model_manager.test(test_loader)
        logger.info("评估结果:")
        for metric_name, value in test_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        return
    
    # 开始训练
    logger.info("开始训练...")
    try:
        model_manager.train(
            train_loader, 
            val_loader,
            use_amp=args.amp
        )
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise
    
    # 在测试集上评估
    logger.info("在测试集上评估...")
    test_metrics = model_manager.test(test_loader)
    logger.info("测试结果:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # 绘制训练过程的指标
    logger.info("绘制训练指标...")
    plotter = MetricsPlotter(save_dir=model_manager.model_dir / 'plots')
    plotter.plot_metrics(model_manager.metrics)
    plotter.plot_confusion_matrices(model_manager.conf_matrices)
    
    # 清理资源
    model_manager.cleanup()
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main() 