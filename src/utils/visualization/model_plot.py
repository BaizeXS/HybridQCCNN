from pathlib import Path
from typing import Callable, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

class ModelPlotter:
    """Model Plotter
    
    This class provides methods for plotting activation functions and other model-related graphics.
    
    Attributes:
        None
    """
    
    def plot_activation_function(
        self,
        func: Callable,
        name: str,
        x_range: Tuple[float, float] = (-5, 5),
        num_points: int = 1000,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Plot activation function curve
        
        Args:
            func (Callable): The activation function to plot
            name (str): The name of the activation function
            x_range (Tuple[float, float]): The range of the x-axis, default is (-5, 5)
            num_points (int): The number of points on the x-axis, default is 1000
            save_path (Optional[Union[str, Path]]): The path to save the plot, if None, the plot will not be saved
            show (bool): Whether to display the plot, default is True
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = func(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"{name} Activation Function")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close() 