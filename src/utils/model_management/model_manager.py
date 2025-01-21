import importlib.util
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import ALL_MODELS
from utils.training.trainer import Trainer


class ModelManager:
    """Model manager: responsible for managing a single model, including building, training, testing, etc.
    
    This class handles model lifecycle including:
    - Model initialization and building
    - Training and validation
    - Checkpointing and model weights management 
    - Metrics tracking and logging
    - TensorBoard visualization
    
    Attributes:
        config (Config): Configuration object containing model and training parameters.
        model_name (str): Name of the model instance.
        device (torch.device): Device to run the model on.
        model (nn.Module): The actual model instance.
        logger (logging.Logger): Logger instance for this model.
        writer (SummaryWriter): TensorBoard writer.
        metrics (dict): Dictionary storing training/validation/test metrics.
        conf_matrices (dict): Dictionary storing confusion matrices.
    """

    def __init__(self, config: Config, model_name: str = "default"):
        """Initialize model manager.
        
        Args:
            config (Config): Configuration object containing model and training parameters.
            model_name (str): Name of the model instance, used to distinguish different instances of the same model.
        """
        # Initialize configuration
        self.config = config
        self.model_name = model_name

        # Set device
        self.device = torch.device(config.device)

        # Initialize directories first
        self._init_directories()

        # Initialize logger
        self.logger = logging.getLogger(f"{self.model_name}_manager")
        self._setup_logging()

        # Set random seed
        if hasattr(self.config, 'seed'):
            self.logger.info(f"Set random seed to {config.seed}")
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            np.random.seed(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Initialize metrics tracking
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.conf_matrices = defaultdict(list)

        # Initialize TensorBoard
        self.writer = SummaryWriter(self.tensorboard_dir)  # type: ignore[arg-type]

    def _init_directories(self):
        """Set up model directory structure."""
        self.model_dir = self.config.base_dir / self.model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"  # Store checkpoints
        self.weights_dir = self.model_dir / "weights"  # Store model weights
        self.tensorboard_dir = self.config.tensorboard_dir / self.model_name  # TensorBoard logs
        self.log_dir = self.model_dir / "logs"  # Log files
        self.metrics_dir = self.model_dir / "metrics"  # Store metrics and confusion matrix data

        # Create necessary directories
        for dir_path in [self.model_dir, self.checkpoint_dir,
                         self.weights_dir, self.tensorboard_dir,
                         self.log_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Set up logging.
        
        This method configures the logger to output messages to both the console and a log file.
        The log file is stored in the model's log directory.
        """
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"{self.model_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _build_model(self):
        """Build model.
        
        This method constructs the model based on the configuration provided. It can load a custom model
        or select from predefined models.
        
        Returns:
            nn.Module: The constructed model instance.
        
        Raises:
            ValueError: If the model type is unknown.
        """
        if self.config.model.model_type == "custom":
            # Load custom model
            model_path = Path(self.config.model.custom_model_path)
            spec = importlib.util.spec_from_file_location(
                model_path.stem, model_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[model_path.stem] = module
            spec.loader.exec_module(module)

            # Get model class
            model_class = getattr(module, self.config.model.name)
        else:
            model_class = ALL_MODELS.get(self.config.model.name)

        if model_class is None:
            raise ValueError(f"Unknown model type: {self.config.model.name}")

        # Create a copy of model_kwargs to avoid modifying the original configuration
        model_kwargs = self.config.model.model_kwargs.copy()

        # Handle num_classes
        if 'num_classes' in model_kwargs:
            if model_kwargs['num_classes'] != self.config.data.num_classes:
                self.logger.warning(
                    f"Model num_classes ({model_kwargs['num_classes']}) "
                    f"does not match dataset ({self.config.data.num_classes}). "
                    f"Using dataset value."
                )
        model_kwargs['num_classes'] = self.config.data.num_classes

        return model_class(**model_kwargs)

    @staticmethod
    def _get_criterion():
        """Get loss function.
        
        Returns:
            nn.Module: The loss function to be used during training.
        """
        return torch.nn.CrossEntropyLoss()

    def _get_optimizer(self):
        """Get optimizer.
        
        Returns:
            torch.optim.Optimizer: The optimizer to be used during training.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

    def _get_scheduler(self):
        """Get learning rate scheduler.
        
        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: The learning rate scheduler, or None if not configured.
        """
        if not self.config.training.scheduler_kwargs:
            return None
        return torch.optim.lr_scheduler.StepLR(  # type: ignore[return-value]
            self._get_optimizer(),
            **self.config.training.scheduler_kwargs
        )

    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            **kwargs
    ) -> None:
        """Train the model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            **kwargs: Additional arguments passed to trainer.
        """
        trainer = Trainer(
            model=self.model,
            criterion=self._get_criterion(),
            optimizer=self._get_optimizer(),
            scheduler=self._get_scheduler(),
            device=self.device,  # type: ignore[arg-type]
            logger=self.logger,
            **kwargs
        )

        best_val_acc = 0.0
        self.logger.info(f"Start training {self.config.training.num_epochs} epochs")

        for epoch in range(self.config.training.num_epochs):
            # Train
            train_metrics, train_conf = trainer.train_epoch(train_loader, epoch)
            self._update_metrics('train', train_metrics, train_conf, epoch)

            # Validate
            if val_loader:
                val_metrics, val_conf = trainer.validate(val_loader)
                self._update_metrics('val', val_metrics, val_conf, epoch)

                # Check if it is the best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch, is_best=True)

            # Save checkpoint periodically
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model.
        
        Args:
            test_loader (DataLoader): DataLoader for test data.
        
        Returns:
            Dict[str, float]: A dictionary of metrics for the test phase.
        """
        self.model.eval()
        trainer = Trainer(
            model=self.model,
            criterion=self._get_criterion(),
            optimizer=None,  # type: ignore[arg-type]
            device=self.device,  # type: ignore[arg-type]
        )
        test_metrics, conf_matrix = trainer.evaluate(test_loader)
        self._update_metrics('test', test_metrics, conf_matrix, epoch=None)  # type: ignore[arg-type]
        return test_metrics

    def predict(self, inputs):
        """Model prediction.
        
        Args:
            inputs (torch.Tensor): Input data for prediction.
        
        Returns:
            torch.Tensor: Predicted class indices.
        """
        self.model.eval()
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint.
        
        Args:
            epoch (int): Current epoch number.
            is_best (bool): Flag indicating if this is the best model.
        """
        # Save model-related data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),  # type: ignore[attr-defined]
            'optimizer_state_dict': self._get_optimizer().state_dict(),  # type: ignore[attr-defined]
        }
        if self._get_scheduler():
            checkpoint['scheduler_state_dict'] = self._get_scheduler().state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model if it is the best
        if is_best:
            best_path = self.weights_dir / 'best_model.pt'
            torch.save(self.model.state_dict(), best_path)
            self.logger.info(f"Save best model weights to {best_path}")

        # Save current epoch metrics
        current_metrics = {
            'metrics': {
                phase: {
                    metric_name: values[-1] if values else None  # 只取最后一个值
                    for metric_name, values in phase_metrics.items()
                }
                for phase, phase_metrics in self.metrics.items()
            },
            'conf_matrices': {
                phase: matrices[-1].tolist() if matrices else None  # 只取最后一个矩阵
                for phase, matrices in self.conf_matrices.items()
            },
            'epoch': epoch
        }
        metrics_path = self.metrics_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(current_metrics, f, indent=2)  # type: ignore[arg-type]

        # Save history metrics
        history_metrics = {
            'metrics': {
                phase: {
                    metric_name: np.array(values).tolist()
                    for metric_name, values in phase_metrics.items()
                }
                for phase, phase_metrics in self.metrics.items()
            },
            'conf_matrices': {
                phase: [matrix.tolist() for matrix in matrices]
                for phase, matrices in self.conf_matrices.items()
            },
            'epoch': epoch
        }
        history_path = self.metrics_dir / 'metrics_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_metrics, f, indent=2)  # type: ignore[arg-type]

        self.logger.info(f"Save checkpoint to {checkpoint_path}")
        self.logger.info(f"Save metrics to {metrics_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint.
        
        Args:
            checkpoint_path (Union[str, Path]): Path to the checkpoint file.
            
        Returns:
            int: The epoch number of the loaded checkpoint.
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )

            # Validate checkpoint structure
            required_keys = {'epoch', 'model_state_dict'}
            if not all(key in checkpoint for key in required_keys):
                raise RuntimeError("Checkpoint file is corrupted or invalid")

            self.model.load_state_dict(checkpoint['model_state_dict'])  # type: ignore[attr-defined]
            if 'optimizer_state_dict' in checkpoint:
                self._get_optimizer().load_state_dict(checkpoint['optimizer_state_dict'])  # type: ignore[attr-defined]
            if 'scheduler_state_dict' in checkpoint and self._get_scheduler():
                self._get_scheduler().load_state_dict(checkpoint['scheduler_state_dict'])

            return checkpoint['epoch']
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def load_metrics(self, epoch: Optional[int] = None, metrics_path: Optional[Path] = None):
        """Load metrics data.
        
        Args:
            epoch (Optional[int]): Epoch number. If provided, load metrics for specific epoch.
                                     If None, load complete metrics history.
            metrics_path (Optional[Path]): Optional path to metrics file. 
                                         If not provided, will look in default location.
        
        Returns:
            int: The epoch number of the loaded metrics.
            
        Raises:
            FileNotFoundError: If metrics file not found.
        """
        if metrics_path is None:
            if epoch is not None:
                # Load specific epoch metrics
                metrics_path = self.metrics_dir / f"metrics_epoch_{epoch}.json"
            else:
                # Load complete history
                metrics_path = self.metrics_dir / "metrics_history.json"
        else:
            metrics_path = Path(metrics_path)

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        self.logger.info(f"Load metrics data: {metrics_path}")
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        self.metrics = defaultdict(lambda: defaultdict(list))

        if epoch is not None:
            # For single epoch metrics, convert single values to lists
            for phase, phase_metrics in metrics_data['metrics'].items():
                for metric_name, value in phase_metrics.items():
                    self.metrics[phase][metric_name] = [value] if value is not None else []

            # Convert single confusion matrix to list
            self.conf_matrices = {
                phase: [np.array(matrix)] if matrix is not None else []
                for phase, matrix in metrics_data['conf_matrices'].items()
            }
        else:
            # For complete history, load as is
            for phase, phase_metrics in metrics_data['metrics'].items():
                for metric_name, values in phase_metrics.items():
                    self.metrics[phase][metric_name] = values

            self.conf_matrices = {
                phase: [np.array(matrix) for matrix in matrices]
                for phase, matrices in metrics_data['conf_matrices'].items()
            }

        return metrics_data['epoch']

    def load_weights(self, weights_path: Union[str, Path]):
        """Load model weights only.
        
        Args:
            weights_path (Union[str, Path]): Path to the weights file.
            
        Raises:
            FileNotFoundError: If weights file doesn't exist.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self.logger.info(f"Load model weights: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _update_metrics(self, phase, metrics, conf_matrix, epoch):
        """Update and record metrics.
        
        Args:
            phase (str): The phase of training (train, val, test).
            metrics (Dict[str, float]): The metrics to update.
            conf_matrix (np.ndarray): The confusion matrix for the current phase.
            epoch (int): The current epoch number.
        """
        # Update metrics history
        for name, value in metrics.items():
            self.metrics[phase][name].append(value)
            self.writer.add_scalar(f'{phase}/{name}', value, epoch)

        # Save confusion matrix
        self.conf_matrices[phase].append(conf_matrix)

    def get_model_summary(self) -> Dict:
        """Get model summary including parameters count and architecture.
        
        Returns:
            Dict: A dictionary containing model information such as model name, type, total parameters, and trainable parameters.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'model_type': self.config.model.name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }

    def cleanup(self):
        """Cleanup resources like TensorBoard writer.
        
        This method closes the TensorBoard writer if it exists.
        """
        if hasattr(self, 'writer'):
            self.writer.close()
