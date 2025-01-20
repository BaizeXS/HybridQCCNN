import pytest
import numpy as np
import torch
import pennylane as qml
from pathlib import Path
from utils.visualization import MetricsPlotter, QuantumPlotter, ModelPlotter
import json

@pytest.fixture
def metrics_data():
    """Return metrics data for testing"""
    return {
        'train': {
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.8, 0.85, 0.9]
        },
        'val': {
            'loss': [0.55, 0.45, 0.35],
            'accuracy': [0.75, 0.8, 0.85]
        }
    }

@pytest.fixture
def confusion_matrix():
    """Return confusion matrix for testing"""
    return np.array([
        [90, 10],
        [5, 95]
    ])

@pytest.fixture
def quantum_circuit():
    """Return quantum circuit for testing"""
    dev = qml.device('default.qubit', wires=2)
    @qml.qnode(dev)
    def circuit(x, weights):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    return circuit

@pytest.fixture
def multi_model_metrics():
    """Return metrics data for multiple models"""
    return {
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

@pytest.fixture
def metrics_history_file(tmp_path):
    """Create a temporary metrics history file for testing"""
    metrics_data = {
        'metrics': {
            'train': {
                'loss': [0.5, 0.4, 0.3],
                'accuracy': [0.8, 0.85, 0.9]
            },
            'val': {
                'loss': [0.55, 0.45, 0.35],
                'accuracy': [0.75, 0.8, 0.85]
            }
        },
        'conf_matrices': {
            'train': [[90, 10], [5, 95]],
            'val': [[85, 15], [10, 90]]
        }
    }
    
    file_path = tmp_path / "metrics_history.json"
    with open(file_path, 'w') as f:
        json.dump(metrics_data, f)
    return file_path

class TestMetricsPlotter:
    """Test MetricsPlotter class"""
    
    def test_plot_single_metric(self, metrics_data, tmp_path):
        """Test single metric plotting functionality"""
        plotter = MetricsPlotter()
        save_path = tmp_path / "loss.png"
        
        plotter.plot_single_metric(
            data=metrics_data,
            metric_name='loss',
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()
    
    def test_plot_confusion_matrix(self, confusion_matrix, tmp_path):
        """Test confusion matrix plotting functionality"""
        plotter = MetricsPlotter()
        save_path = tmp_path / "confusion_matrix.png"
        
        plotter.plot_confusion_matrix(
            conf_matrix=confusion_matrix,
            classes=['Class 0', 'Class 1'],
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()

    def test_plot_metrics_comparison(self, multi_model_metrics, tmp_path):
        """Test metrics comparison plotting functionality"""
        plotter = MetricsPlotter()
        save_dir = tmp_path / "comparison_plots"
        
        plotter.plot_metrics_comparison(
            data=multi_model_metrics,
            metric_names=['loss', 'accuracy'],
            model_names=['model1', 'model2'],
            phases=['train', 'val'],
            save_dir=save_dir,
            show=False
        )
        
        # Validate the existence of the generated plots
        assert (save_dir / "loss_comparison.png").exists()
        assert (save_dir / "accuracy_comparison.png").exists()

    def test_plot_metrics_comparison_subset(self, multi_model_metrics, tmp_path):
        """Test metrics comparison with subset of models and phases"""
        plotter = MetricsPlotter()
        save_dir = tmp_path / "subset_plots"
        
        plotter.plot_metrics_comparison(
            data=multi_model_metrics,
            metric_names=['loss'],
            model_names=['model1'],
            phases=['train'],
            save_dir=save_dir,
            show=False
        )
        
        assert (save_dir / "loss_comparison.png").exists()

    def test_plot_from_saved_metrics(self, metrics_history_file, tmp_path):
        """Test plotting from saved metrics file"""
        plotter = MetricsPlotter()
        save_dir = tmp_path / "saved_metrics_plots"
        
        plotter.plot_from_saved_metrics(
            metrics_path=metrics_history_file,
            metric_names=['loss', 'accuracy'],
            phases=['train', 'val'],
            save_dir=save_dir,
            show=False
        )
        
        # Validate the existence of the generated plots
        assert (save_dir / "loss_curves.png").exists()
        assert (save_dir / "accuracy_curves.png").exists()
        assert (save_dir / "confusion_matrix_train.png").exists()
        assert (save_dir / "confusion_matrix_val.png").exists()

    def test_plot_from_saved_metrics_invalid_file(self, tmp_path):
        """Test handling of invalid metrics file"""
        plotter = MetricsPlotter()
        invalid_file = tmp_path / "invalid_metrics.json"
        
        with pytest.raises(FileNotFoundError):
            plotter.plot_from_saved_metrics(
                metrics_path=invalid_file,
                save_dir=tmp_path,
                show=False
            )

    def test_plot_from_saved_metrics_invalid_format(self, tmp_path):
        """Test handling of invalid metrics file format"""
        invalid_data = {'invalid': 'format'}
        invalid_file = tmp_path / "invalid_format.json"
        
        with open(invalid_file, 'w') as f:
            json.dump(invalid_data, f)
        
        plotter = MetricsPlotter()
        with pytest.raises(ValueError, match="Invalid metrics file format"):
            plotter.plot_from_saved_metrics(
                metrics_path=invalid_file,
                save_dir=tmp_path,
                show=False
            )

    def test_plot_single_metric_invalid_data(self):
        """Test handling of invalid data in single metric plotting"""
        plotter = MetricsPlotter()
        invalid_data = {
            'train': {
                'loss': 'not_a_list'
            }
        }
        
        with pytest.raises(TypeError, match="must be a list"):
            plotter.plot_single_metric(
                data=invalid_data,
                metric_name='loss'
            )

    def test_plot_confusion_matrix_invalid_input(self):
        """Test handling of invalid confusion matrix input"""
        plotter = MetricsPlotter()
        invalid_matrix = [[1, 2], [3]]
        
        with pytest.raises(ValueError):
            plotter.plot_confusion_matrix(
                conf_matrix=np.array(invalid_matrix),
                classes=['A', 'B']
            )

class TestQuantumPlotter:
    """Test QuantumPlotter class"""
    
    def test_plot_quantum_state(self, tmp_path):
        """Test quantum state plotting functionality"""
        plotter = QuantumPlotter()
        save_path = tmp_path / "quantum_state.png"
        state = np.array([0, 0, 1])
        
        plotter.plot_quantum_state(
            state=state,
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()
    
    def test_plot_quantum_circuit(self, quantum_circuit, tmp_path):
        """Test quantum circuit plotting functionality"""
        plotter = QuantumPlotter()
        save_path = tmp_path / "quantum_circuit.png"
        
        plotter.plot_quantum_circuit(
            qnode=quantum_circuit,
            inputs=np.array([0.5, 0.1]),
            weights=None,
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()

class TestModelPlotter:
    """Test ModelPlotter class"""
    
    def test_plot_activation_function(self, tmp_path):
        """Test activation function plotting functionality"""
        plotter = ModelPlotter()
        save_path = tmp_path / "activation.png"
        
        def relu(x):
            return np.maximum(0, x)
        
        plotter.plot_activation_function(
            func=relu,
            name='ReLU',
            save_path=save_path,
            show=False
        )
        
        assert save_path.exists()
    
    @pytest.mark.parametrize("x_range", [
        (-5, 5),
        (-10, 10),
        (0, 1)
    ])
    def test_activation_function_ranges(self, x_range):
        """Test activation function plotting with different input ranges"""
        plotter = ModelPlotter()
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        plotter.plot_activation_function(
            func=sigmoid,
            name='Sigmoid',
            x_range=x_range,
            show=False
        ) 