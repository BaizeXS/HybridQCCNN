import torch
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class MetricsCalculator:
    """Metrics calculator: responsible for calculating various training and evaluation metrics"""
    
    @staticmethod
    def calculate(outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> Tuple[Dict[str, float], np.ndarray]:
        """Calculate metrics for a batch of data
        
        Args:
            outputs: Model outputs
            targets: True labels
            loss: Loss value
            
        Returns:
            A dictionary containing various metrics and a confusion matrix
        """
        # Get prediction results
        _, predictions = torch.max(outputs, 1)
        
        # Convert to numpy array for sklearn
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Get the number of classes
        num_classes = outputs.size(1)
        
        # Calculate basic metrics
        metrics = {
            'loss': loss.item(),
            'accuracy': (predictions == targets).mean(),
            'precision': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall': recall_score(targets, predictions, average='macro', zero_division=0),
            'f1': f1_score(targets, predictions, average='macro', zero_division=0)
        }
        
        # Calculate confusion matrix, ensuring correct number of classes
        conf_matrix = confusion_matrix(
            targets, 
            predictions, 
            labels=range(num_classes)
        )
        
        return metrics, conf_matrix 