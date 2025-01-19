import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class MetricsCalculator:
    """Metrics calculator for evaluating model performance.
    
    This class provides methods to calculate various metrics such as loss, accuracy,
    precision, recall, F1 score, and confusion matrix for a given set of model outputs
    and true labels. It is designed to be used during training and evaluation phases
    of machine learning models.
    
    Attributes:
        None
    """
    
    @staticmethod
    def calculate(outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> Tuple[Dict[str, float], np.ndarray]:
        """Calculate metrics for a batch of data.
        
        Args:
            outputs (torch.Tensor): The raw output predictions from the model.
            targets (torch.Tensor): The true labels corresponding to the outputs.
            loss (torch.Tensor): The computed loss value for the batch.
            
        Returns:
            Tuple[Dict[str, float], np.ndarray]: A tuple containing:
                - A dictionary with keys 'loss', 'accuracy', 'precision', 'recall', and 'f1',
                  representing the calculated metrics for the batch.
                - A confusion matrix as a NumPy array, representing the performance of the model
                  in classifying the input data.
        """
        # Get prediction results
        _, predictions = torch.max(outputs, dim=1)
        
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