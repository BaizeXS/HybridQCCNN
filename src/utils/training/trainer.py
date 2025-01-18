import torch
from collections import defaultdict
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
import time
from .metrics import MetricsCalculator

class Trainer:
    """Trainer"""
    
    def __init__(
        self, 
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        aux_weight: float = 0.4,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize Trainer
        
        Args:
            model: Model
            criterion: Loss function
            optimizer: Optimizer
            device: Device
            scheduler: Learning rate scheduler
            aux_weight: GoogLeNet auxiliary classifier weight
            logger: Logger
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.aux_weight = aux_weight
        self.metrics_calculator = MetricsCalculator()
        self.logger = logger or logging.getLogger(__name__)
        # Check if the model is a GoogLeNet model
        self.is_googlenet = 'GoogLeNet' in model.__class__.__name__


    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[Dict[str, float], np.ndarray]:
        """Train one epoch"""
        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        conf_matrix = None
        
        epoch_start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=True)
        
        for batch in pbar:
            batch_start_time = time.time()
            
            batch_metrics, batch_conf_matrix = self._train_step(batch)
            batch_size = batch[0].size(0)
            total_samples += batch_size
            
            batch_time = time.time() - batch_start_time
            
            # Accumulate batch metrics
            for name, value in batch_metrics.items():
                metrics[name] += value * batch_size
                
            # Update confusion matrix
            if conf_matrix is None:
                conf_matrix = batch_conf_matrix
            else:
                conf_matrix += batch_conf_matrix
                
            # Update progress bar, add time information
            pbar.set_postfix({
                'loss': batch_metrics['loss'],
                'acc': batch_metrics['accuracy'],
                'batch_time': f'{batch_time:.3f}s'
            })
        
        # Calculate average metrics
        avg_metrics = {name: value/total_samples for name, value in metrics.items()}
        
        # Calculate the time of the entire epoch
        epoch_time = time.time() - epoch_start_time
        
        # Log the epoch metrics, add time information
        self.logger.info(
            f"Train Epoch {epoch} - " + 
            " - ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]) +
            f" - epoch_time: {epoch_time:.2f}s"
        )
        
        # Update learning rate at the end of the epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Add time information to metrics
        avg_metrics['epoch_time'] = epoch_time
        
        return avg_metrics, conf_matrix

    def validate(self, dataloader: DataLoader, phase: str = 'val') -> Tuple[Dict[str, float], np.ndarray]:
        """Validate model"""
        self.model.eval()
        metrics = defaultdict(float)
        total_samples = 0
        conf_matrix = None
        
        phase_start_time = time.time()
        pbar = tqdm(dataloader, desc=f'{phase.capitalize()} Phase', leave=True)
        
        with torch.no_grad():
            for batch in pbar:
                batch_start_time = time.time()
                
                batch_metrics, batch_conf_matrix = self._validate_step(batch)
                batch_size = batch[0].size(0)
                total_samples += batch_size
                
                batch_time = time.time() - batch_start_time
                
                for name, value in batch_metrics.items():
                    metrics[name] += value * batch_size
                    
                if conf_matrix is None:
                    conf_matrix = batch_conf_matrix
                else:
                    conf_matrix += batch_conf_matrix
                    
                pbar.set_postfix({
                    'loss': batch_metrics['loss'],
                    'acc': batch_metrics['accuracy'],
                    'batch_time': f'{batch_time:.3f}s'
                })
        
        avg_metrics = {name: value/total_samples for name, value in metrics.items()}
        
        # Calculate the time of the entire validation phase
        phase_time = time.time() - phase_start_time
        
        self.logger.info(
            f"{phase.capitalize()} Phase - " + 
            " - ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]) +
            f" - time: {phase_time:.2f}s"
        )
        
        # Add time information to metrics
        avg_metrics['phase_time'] = phase_time
        
        return avg_metrics, conf_matrix

    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model"""
        return self.validate(dataloader, phase='test')
    
    def _train_step(self, batch) -> Tuple[Dict[str, float], np.ndarray]:
        """Single training step"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward propagation
        self.optimizer.zero_grad()
        # For GoogLeNet model, process auxiliary classifier output
        if self.is_googlenet and self.model.training:
            output, aux_output = self.model(inputs)
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(aux_output, targets)
            loss = loss1 + self.aux_weight * loss2
        else:
            output = self.model(inputs)
            loss = self.criterion(output, targets)
        
        # Backward propagation
        loss.backward()
        self.optimizer.step()
        
        return self.metrics_calculator.calculate(output, targets, loss)

    def _validate_step(self, batch) -> Tuple[Dict[str, float], np.ndarray]:
        """Single validation step"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.model(inputs)
        # For GoogLeNet model, use only the main output during validation
        if self.is_googlenet:
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = self.criterion(outputs, targets)

        return self.metrics_calculator.calculate(outputs, targets, loss)
