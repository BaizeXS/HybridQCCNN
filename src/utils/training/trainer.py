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
    """A trainer class that handles model training, validation and evaluation.
    
    This class provides functionality for:
    - Training models epoch by epoch
    - Validating model performance
    - Evaluating model on test data
    - Handling different model architectures (including GoogLeNet with auxiliary classifiers)
    - Computing and logging metrics
    - Progress tracking with tqdm
    
    Attributes:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): Loss function for model training.
        optimizer (torch.optim.Optimizer): Optimization algorithm for training.
        device (str): Device to run the model on ('cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        aux_weight (float): Weight for auxiliary classifiers in GoogLeNet.
        metrics_calculator (MetricsCalculator): Calculator for various metrics.
        logger (logging.Logger): Logger for tracking training progress.
        is_googlenet (bool): Flag indicating if the model is GoogLeNet.
        memory_tracking (bool): Whether to track GPU memory usage.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        aux_weight: float = 0.4,
        logger: Optional[logging.Logger] = None,
        memory_tracking: bool = False
    ):
        """Initialize Trainer.
        
        Args:
            model (torch.nn.Module): Neural network model to be trained.
            criterion (torch.nn.Module): Loss function for model training.
            optimizer (torch.optim.Optimizer): Optimization algorithm for training.
            device (str): Device to run the model on ('cpu' or 'cuda').
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            aux_weight (float): Weight for GoogLeNet auxiliary classifiers. Defaults to 0.4.
            logger (logging.Logger, optional): Logger for tracking training progress.
            memory_tracking (bool): Whether to track GPU memory usage. Defaults to False.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.aux_weight = aux_weight
        self.metrics_calculator = MetricsCalculator()
        self.logger = logger or logging.getLogger(__name__)
        self.memory_tracking = memory_tracking
        # Check if the model is a GoogLeNet model
        self.is_googlenet = 'GoogLeNet' in model.__class__.__name__

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[Dict[str, float], np.ndarray]:
        """Train the model for one epoch.
        
        Args:
            dataloader (DataLoader): DataLoader containing training data.
            epoch (int): Current epoch number.
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary with average metrics for the epoch, including loss, accuracy, precision, recall, and F1 score.
                - A confusion matrix as a NumPy array for the epoch.
        """
        self.model.train()
        metrics = defaultdict(float)
        total_samples = 0
        conf_matrix = None
        
        # Start time of the epoch
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
        """Validate or evaluate the model on a given dataset.
        
        Args:
            dataloader (DataLoader): DataLoader containing validation/test data.
            phase (str, optional): Phase identifier ('val' or 'test'). Defaults to 'val'.
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary of averaged metrics for the validation/test phase.
                - A confusion matrix as a NumPy array for the entire validation/test set.
        """
        self.model.eval()
        metrics = defaultdict(float)
        total_samples = 0
        conf_matrix = None
        
        # Start time of the validation phase
        phase_start_time = time.time()
        
        # Create progress bar
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
        
        # Calculate average metrics
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
        """Evaluate the model on test dataset.
        
        This is a wrapper around validate() method that sets phase to 'test'.
        Used for final model evaluation after training is complete.
        
        Args:
            dataloader (DataLoader): DataLoader containing test data.
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary of averaged metrics for the test phase.
                - A confusion matrix as a NumPy array for the test set.
        """
        return self.validate(dataloader, phase='test')
    
    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], np.ndarray]:
        """Perform a single training step.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of (inputs, targets) tensors.
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary of metrics (loss, accuracy, etc.) for the current batch.
                - A confusion matrix as a NumPy array for this batch.
        """
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

    def _validate_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], np.ndarray]:
        """Perform a single validation/evaluation step.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of (inputs, targets) tensors.
                
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary of batch metrics (loss, accuracy, etc.).
                - A confusion matrix as a NumPy array for this batch.
                
        Note:
            For GoogLeNet models, only the main classifier output is used during validation.
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.model(inputs)
        # For GoogLeNet model, use only the main output during validation
        if self.is_googlenet:
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = self.criterion(outputs, targets)

        return self.metrics_calculator.calculate(outputs, targets, loss)

    def _log_memory_stats(self):
        """Log GPU memory statistics if memory tracking is enabled.
        
        This method will log:
            - Allocated memory: Currently allocated GPU memory in MB.
            - Cached memory: Total cached (reserved) GPU memory in MB.
            
        Note:
            Only logs if self.memory_tracking is True and CUDA is available.
        """
        if self.memory_tracking and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(f"GPU Memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached")

    def cleanup(self):
        """Clean up resources used by the trainer.
        
        Performs the following cleanup operations:
            1. Clears CUDA cache if GPU was used.
            2. Removes circular references to prevent memory leaks.
            3. Sets model, optimizer, and scheduler to None.
            
        Note:
            This method should be called when the trainer is no longer needed
            or before creating a new trainer instance.
        """
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove circular references
        self.model = None
        self.optimizer = None
        self.scheduler = None
