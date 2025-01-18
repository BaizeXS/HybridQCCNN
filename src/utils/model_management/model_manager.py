import torch
import os
import logging
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from utils.training.trainer import Trainer
import importlib.util
import sys
from pathlib import Path
import numpy as np
from typing import Union, Optional, Dict
from models import ALL_MODELS

class ModelManager:
    """Model manager: responsible for managing a single model, including building, training, testing, etc."""
    
    def __init__(self, config, model_name="default"):
        """Initialize model manager
        
        Args:
            config: Configuration object
            model_name: Name of the model instance, used to distinguish different instances of the same model
        """
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)
        
        # Set up model directory structure
        self.model_dir = config.base_dir / model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"  # Store checkpoints
        self.weights_dir = self.model_dir / "weights"         # Store model weights
        self.tensorboard_dir = config.tensorboard_dir / model_name  # tensorboard logs
        self.log_dir = self.model_dir / "logs"               # Log files
        self.metrics_dir = self.model_dir / "metrics"  # Store metrics and confusion matrix data
        
        # Create necessary directories
        for dir_path in [self.model_dir, self.checkpoint_dir, 
                        self.weights_dir, self.tensorboard_dir, 
                        self.log_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.model_name}_manager")
        self._setup_logging()
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Add TensorBoard support
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Add metrics recording
        self.metrics = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        
        self.conf_matrices = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def _setup_logging(self):
        """Set up logging"""
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
        """Build model"""
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
            
        return model_class(**self.config.model.model_kwargs)

    def _get_criterion(self):
        """Get loss function"""
        return torch.nn.CrossEntropyLoss()

    def _get_optimizer(self):
        """Get optimizer"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

    def _get_scheduler(self):
        """Get learning rate scheduler"""
        if not self.config.training.scheduler_kwargs:
            return None
        return torch.optim.lr_scheduler.StepLR(
            self._get_optimizer(),
            **self.config.training.scheduler_kwargs
        )

    def train(self, train_loader, val_loader=None, **kwargs):
        """Train model"""
        trainer = Trainer(
            model=self.model,
            criterion=self._get_criterion(),
            optimizer=self._get_optimizer(),
            scheduler=self._get_scheduler(),
            device=self.device,
            logger=self.logger
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

    def test(self, test_loader):
        """Test model"""
        self.model.eval()
        trainer = Trainer(
            model=self.model,
            criterion=self._get_criterion(),
            optimizer=self._get_optimizer(),
            device=self.device
        )
        test_metrics, conf_matrix = trainer.evaluate(test_loader)
        self._update_metrics('test', test_metrics, conf_matrix, epoch=None)
        return test_metrics

    def predict(self, inputs):
        """Model prediction"""
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        # Save model-related data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self._get_optimizer().state_dict(),
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
        
        # Save metrics and confusion matrix separately
        metrics_data = {
            'metrics': self.metrics,
            'conf_matrices': self.conf_matrices,
            'epoch': epoch
        }
        metrics_path = self.metrics_dir / f'metrics_epoch_{epoch}.pt'
        torch.save(metrics_data, metrics_path)
        
        self.logger.info(f"Save checkpoint to {checkpoint_path}")
        self.logger.info(f"Save metrics to {metrics_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        self.logger.info(f"Load checkpoint: {checkpoint_path}")
        # Use weights_only=True for added security
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=True  # Add this parameter
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self._get_optimizer().load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self._get_scheduler():
            self._get_scheduler().load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']

    def load_metrics(self, epoch: int):
        """Load metrics data for a specific epoch"""
        metrics_path = self.metrics_dir / f'metrics_epoch_{epoch}.pt'
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
            
        self.logger.info(f"Load metrics data: {metrics_path}")
        metrics_data = torch.load(metrics_path)
        self.metrics = metrics_data['metrics']
        self.conf_matrices = metrics_data['conf_matrices']
        return metrics_data['epoch']

    def load_weights(self, weights_path: Union[str, Path]):
        """Load model weights only"""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        self.logger.info(f"Load model weights: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _update_metrics(self, phase, metrics, conf_matrix, epoch):
        """Update and record metrics"""
        # Update metrics history
        for name, value in metrics.items():
            self.metrics[phase][name].append(value)
            self.writer.add_scalar(f'{phase}/{name}', value, epoch)
            
        # Save confusion matrix
        self.conf_matrices[phase].append(conf_matrix) 