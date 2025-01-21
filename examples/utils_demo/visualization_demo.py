from pathlib import Path

import numpy as np
import pennylane as qml

from utils.visualization import MetricsPlotter, QuantumPlotter, ModelPlotter


def metrics_plotter_example(output_dir: Path):
    """Demonstrate MetricsPlotter functionality
    
    Shows how to:
    1. Plot single metric with multiple phases
    2. Plot confusion matrix
    3. Plot metrics comparison across models
    """
    print("MetricsPlotter example:")

    # Create example training metrics
    metrics_data = {
        'train': {
            'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'accuracy': [0.8, 0.85, 0.9, 0.92, 0.95]
        },
        'val': {
            'loss': [0.55, 0.45, 0.35, 0.25, 0.15],
            'accuracy': [0.75, 0.8, 0.85, 0.87, 0.9]
        }
    }

    # Create example confusion matrix
    conf_matrix = np.array([
        [90, 10],
        [5, 95]
    ])

    # Create example multi-model metrics
    multi_model_metrics = {
        'model1': {
            'train': {
                'loss': [0.5, 0.4, 0.3],
                'accuracy': [0.8, 0.85, 0.9]
            },
            'val': {
                'loss': [0.55, 0.45, 0.35],
                'accuracy': [0.75, 0.8, 0.85]
            }
        },
        'model2': {
            'train': {
                'loss': [0.45, 0.35, 0.25],
                'accuracy': [0.85, 0.9, 0.95]
            },
            'val': {
                'loss': [0.5, 0.4, 0.3],
                'accuracy': [0.8, 0.85, 0.9]
            }
        }
    }

    # Create MetricsPlotter instance
    plotter = MetricsPlotter()

    # 1. Plot single metric
    plotter.plot_single_metric(
        data=metrics_data,
        metric_name='loss',
        title='Training and Validation Loss',
        save_path=output_dir / 'single_metric.png',
        show=True
    )

    # 2. Plot confusion matrix
    plotter.plot_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=['Class 0', 'Class 1'],
        save_path=output_dir / 'confusion_matrix.png',
        show=True
    )

    # 3. Plot metrics comparison
    plotter.plot_metrics_comparison(
        data=multi_model_metrics,
        metric_names=['loss', 'accuracy'],
        model_names=['model1', 'model2'],
        phases=['train', 'val'],
        save_dir=output_dir / 'comparison',
        show=True
    )


def quantum_plotter_example(output_dir: Path):
    """Demonstrate QuantumPlotter functionality
    
    Shows how to:
    1. Plot quantum state on Bloch sphere
    2. Plot quantum circuit diagram
    """
    print("\nQuantumPlotter example:")

    # Create quantum state example
    state = np.array([0, 0, 1])

    # Create simple quantum circuit
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def quantum_circuit(x, weights):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    # Create QuantumPlotter instance
    plotter = QuantumPlotter()

    # 1. Plot quantum state
    plotter.plot_quantum_state(
        state=state,
        save_path=output_dir / 'quantum_state.png',
        show=True
    )

    # 2. Plot quantum circuit
    plotter.plot_quantum_circuit(
        qnode=quantum_circuit,
        inputs=np.array([0.5, 0.1]),
        weights=None,
        save_path=output_dir / 'quantum_circuit.png',
        show=True
    )


def model_plotter_example(output_dir: Path):
    """Demonstrate ModelPlotter functionality
    
    Shows how to:
    1. Plot different activation functions
    2. Plot with different input ranges
    """
    print("\nModelPlotter example:")

    # Create ModelPlotter instance
    plotter = ModelPlotter()

    # 1. Plot ReLU activation
    def relu(x):
        return np.maximum(0, x)

    plotter.plot_activation_function(
        func=relu,
        name='ReLU',
        save_path=output_dir / 'relu.png',
        show=True
    )

    # 2. Plot Sigmoid activation with different ranges
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for x_range in [(-5, 5), (-10, 10), (0, 1)]:
        plotter.plot_activation_function(
            func=sigmoid,
            name=f'Sigmoid (range: {x_range})',
            x_range=x_range,
            save_path=output_dir / f'sigmoid_range_{x_range[0]}_{x_range[1]}.png',
            show=True
        )


def main():
    """Run all visualization examples"""
    # Create output directory
    output_dir = Path("outputs/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run examples
    metrics_plotter_example(output_dir)
    quantum_plotter_example(output_dir)
    model_plotter_example(output_dir)

    print(f"\nAll plots have been saved to: {output_dir}")


if __name__ == "__main__":
    main()
