from pathlib import Path
from typing import Optional, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from qutip import Bloch


class QuantumPlotter:
    """Quantum Plotter
    
    This class provides methods for plotting quantum states and quantum circuit diagrams.
    
    Attributes:
        None
    """

    @staticmethod
    def plot_quantum_state(
            state: Optional[np.ndarray] = None,
            save_path: Optional[Union[str, Path]] = None,
            show: bool = True
    ) -> None:
        """Plot quantum state on Bloch sphere
        
        Args:
            state (Optional[np.ndarray]): The quantum state to plot, default is None
            save_path (Optional[Union[str, Path]]): The path to save the plot, if None, the plot will not be saved
            show (bool): Whether to display the plot, default is True
        """
        sphere = Bloch()
        sphere.frame_color = 'gray'
        sphere.font_size = 16
        sphere.sphere_color = 'lightblue'
        sphere.vector_width = 2
        sphere.figsize = [9, 9]

        sphere.vector_color = ['r', 'g', 'b', 'purple'] if state is not None else ['r', 'g', 'b']

        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if state is not None:
            vectors.append(state)

        for vector in vectors:
            sphere.add_vectors(vector)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sphere.save(str(save_path))
        if show:
            sphere.show()
        sphere.clear()

    @staticmethod
    def plot_quantum_circuit(
            qnode: qml.QNode,
            inputs: Any,
            weights: Any,
            style: str = "pennylane",
            save_path: Optional[Union[str, Path]] = None,
            show: bool = True
    ) -> None:
        """Plot quantum circuit diagram
        
        Args:
            qnode (qml.QNode): The quantum node to plot
            inputs (Any): The input to the quantum circuit
            weights (Any): The weights of the quantum circuit
            style (str): The style of the plot, default is "pennylane"
            save_path (Optional[Union[str, Path]]): The path to save the plot, if None, the plot will not be saved
            show (bool): Whether to display the plot, default is True
        """
        qml.drawer.use_style(style)
        fig, _ = qml.draw_mpl(qnode)(inputs, weights)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()
