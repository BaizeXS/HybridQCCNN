import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MetricsPlotter:
    """Metrics visualization tool for model evaluation and comparison.

    This class provides methods for visualizing various training metrics and model
    comparisons. It supports:
    1. Single/multiple model metrics visualization
    2. Single/multiple metric display
    3. Single/multiple phase (train/val/test) comparison
    4. Direct data input or file loading
    5. Flexible comparison options

    The plotter can generate:
    - Individual metric curves
    - Model comparison plots
    - Confusion matrices

    Each plot can be saved to a file and/or displayed interactively.

    Attributes:
        None
    """

    def _setup_plot(
        self,
        title: str,
        xlabel: str = "Epoch",
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """Set basic plot properties.

        Args:
            title (str): The title of the plot.
            xlabel (str, optional): Label for x-axis. Defaults to 'Epoch'.
            ylabel (str, optional): Label for y-axis. If None, uses title.
            figsize (Tuple[int, int], optional): Figure size (width, height).
                Defaults to (10, 6).
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or title)
        plt.grid(True)

    def _save_and_show(
        self, save_path: Optional[Union[str, Path]] = None, show: bool = False
    ) -> None:
        """Save and optionally display the plot.

        Args:
            save_path (Optional[Union[str, Path]]): Path to save the plot.
                If None, plot is not saved.
            show (bool): Whether to display the plot. Defaults to False.
        """
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        plt.close()

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        classes: List[str],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """Plot confusion matrix for model evaluation.

        Args:
            conf_matrix (np.ndarray): The confusion matrix to plot.
            classes (List[str]): List of class names for axis labels.
            save_path (Optional[Union[str, Path]]): Path to save the plot.
                If None, plot is not saved.
            show (bool): Whether to display the plot. Defaults to False.
        """
        conf_matrix = np.array(conf_matrix)
        if len(conf_matrix.shape) != 2 or conf_matrix.shape[0] != conf_matrix.shape[1]:
            raise ValueError("Confusion matrix must be a square 2D array")

        plt.figure(figsize=(10, 8))
        fmt = "d" if np.issubdtype(conf_matrix.dtype, np.integer) else ".2f"

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        self._save_and_show(save_path, show)

    def plot_single_metric(
        self,
        data: Dict[str, Dict[str, List[float]]],
        metric_name: str,
        phases: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """Plot a single metric curve for one or more phases.

        This method creates a line plot showing how a specific metric changes over epochs
        for different phases (e.g., training, validation).

        Args:
            data (Dict[str, Dict[str, List[float]]]): Dictionary containing metrics data
                Format: {phase: {metric_name: values}}
            metric_name (str): Name of the metric to plot
            phases (Optional[List[str]]): List of phases to plot
            title (Optional[str]): Custom plot title
            save_path (Optional[Union[str, Path]]): Path to save the plot
            show (bool): Whether to display the plot

        Raises:
            TypeError: If the metric values are not in list format
        """
        self._setup_plot(
            title=title or f"{metric_name.capitalize()} Over Epochs",
            ylabel=metric_name.capitalize(),
        )

        phases = phases or list(data.keys())
        for phase in phases:
            # Check if the phase exists in the data
            if phase not in data:
                continue

            # Get the data for the phase
            phase_data = data[phase]

            # Check if the phase data is a dictionary
            if not isinstance(phase_data, dict):
                raise TypeError(f"Data for phase '{phase}' must be a dictionary")

            # Check if the metric exists in the phase data
            try:
                values = phase_data[metric_name]
                if not isinstance(values, list):
                    raise TypeError(
                        f"Values for '{phase}/{metric_name}' must be a list"
                    )

                if not all(isinstance(v, (int, float)) for v in values):
                    raise TypeError(
                        f"All values in '{phase}/{metric_name}' must be numeric"
                    )

                plt.plot(values, label=phase, marker="o", markersize=3, alpha=0.7)

            except KeyError:
                print(f"Warning: Metric '{metric_name}' not found in phase '{phase}'")
                continue

        plt.legend()
        self._save_and_show(save_path, show)

    def plot_metrics_comparison(
        self,
        data: Dict[str, Dict[str, List[float]]],
        metric_names: List[str],
        model_names: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """Plot comparison of metrics across multiple models and phases.

        This method creates comparison plots for specified metrics across different models
        and phases. Each metric gets its own plot showing the performance of different
        models in different phases.

        Args:
            data (Dict[str, Dict[str, List[float]]]): Nested dictionary containing metrics data.
                Format: {model_name: {phase: {metric_name: values}}}
            metric_names (List[str]): List of metric names to plot (e.g., ['loss', 'accuracy']).
            model_names (Optional[List[str]]): List of model names to include in comparison.
                If None, includes all models in data.
            phases (Optional[List[str]]): List of phases to include in comparison.
                If None, includes all phases available.
            save_dir (Optional[Union[str, Path]]): Directory to save plots.
                If None, uses current working directory.
            show (bool): Whether to display the plots. Defaults to False.
        """
        # Set up save directory
        save_dir = Path(save_dir) if save_dir else Path.cwd()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get model names
        model_names = model_names or list(data.keys())
        if not model_names:
            print("Warning: No models found in data")
            return

        # Iterate over each metric
        for metric_name in metric_names:
            self._setup_plot(
                title=f"{metric_name.capitalize()} Comparison",
                figsize=(12, 6),
                ylabel=metric_name.capitalize(),
            )

            # Plot curves for each model
            for model_name in model_names:
                if model_name not in data:
                    print(f"Warning: Model '{model_name}' not found in data")
                    continue

                model_data = data[model_name]
                if not isinstance(model_data, dict):
                    print(f"Warning: Invalid data format for model '{model_name}'")
                    continue

                curr_phases = phases or list(model_data.keys())

                # Plot curves for each phase
                for phase in curr_phases:
                    try:
                        if phase not in model_data:
                            continue
                        if metric_name not in model_data[phase]:
                            continue

                        values = model_data[phase][metric_name]
                        if not isinstance(values, list):
                            print(
                                f"Warning: Invalid values format for "
                                f"'{model_name}/{phase}/{metric_name}'"
                            )
                            continue

                        plt.plot(
                            values,
                            label=f"{model_name}-{phase}",
                            marker="o",
                            markersize=3,
                            alpha=0.7,
                            linestyle="-",
                        )
                    except Exception as e:
                        print(
                            f"Error plotting {model_name}/{phase}/{metric_name}: {str(e)}"
                        )
                        continue

            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            self._save_and_show(save_dir / f"{metric_name}_comparison.png", show)

    def plot_from_saved_metrics(
        self,
        metrics_path: Union[str, Path],
        metric_names: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """Plot metrics from saved JSON metrics file generated by ModelManager.

        Args:
            metrics_path: Path to the metrics JSON file
            metric_names: List of metric names to plot
            phases: List of phases to plot
            save_dir: Directory to save the plots
            show: Whether to display the plots

        Raises:
            FileNotFoundError: If metrics file doesn't exist
            ValueError: If metrics file format is invalid
        """
        # Load and validate metrics data
        metrics, conf_matrices = self._load_and_validate_metrics(metrics_path)

        # Set up save directory
        metrics_path = Path(metrics_path)
        save_dir = Path(save_dir) if save_dir else metrics_path.parent / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get available phases
        available_phases = set(metrics.keys())
        if phases:
            available_phases = available_phases.intersection(phases)
            if not available_phases:
                raise ValueError(
                    f"No requested phases found. Available phases: {sorted(metrics.keys())}"
                )

        # Get metrics to plot
        metrics_to_plot = self._get_available_metrics(metrics, metric_names)

        # Plot metrics
        for metric_name in metrics_to_plot:
            self.plot_single_metric(
                data=metrics,
                metric_name=metric_name,
                phases=list(available_phases),
                save_path=save_dir / f"{metric_name}_curves.png",
                show=show,
            )

        # Plot confusion matrices
        for phase in available_phases:
            if phase in conf_matrices:
                self._plot_confusion_matrix_for_phase(
                    conf_matrix=conf_matrices[phase],
                    phase=phase,
                    save_path=save_dir / f"confusion_matrix_{phase}.png",
                    show=show,
                )

    def _load_and_validate_metrics(
        self, metrics_path: Union[str, Path]
    ) -> Tuple[Dict, Dict]:
        """Load and validate metrics data from JSON file.

        Args:
            metrics_path: Path to metrics JSON file

        Returns:
            Tuple containing metrics dict and confusion matrices dict

        Raises:
            FileNotFoundError: If metrics file doesn't exist
            ValueError: If metrics file format is invalid
        """
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {metrics_path}")

        if not isinstance(metrics_data, dict):
            raise ValueError("Metrics data must be a dictionary")

        if "metrics" not in metrics_data:
            raise ValueError("Metrics data must contain 'metrics' key")

        return metrics_data["metrics"], metrics_data.get("conf_matrices", {})

    def _get_available_metrics(
        self, metrics: Dict, metric_names: Optional[List[str]] = None
    ) -> List[str]:
        """Get list of metrics to plot based on available and requested metrics.

        Args:
            metrics: Dictionary of metrics data
            metric_names: Optional list of requested metric names

        Returns:
            List of metric names to plot

        Raises:
            ValueError: If requested metrics are not found
        """
        available_metrics = set()
        for phase_metrics in metrics.values():
            available_metrics.update(phase_metrics.keys())

        if metric_names:
            invalid_metrics = set(metric_names) - available_metrics
            if invalid_metrics:
                raise ValueError(
                    f"Requested metrics not found: {invalid_metrics}. "
                    f"Available metrics are: {sorted(available_metrics)}"
                )
            return metric_names

        return sorted(available_metrics)

    def _plot_confusion_matrix_for_phase(
        self,
        conf_matrix: Union[List, np.ndarray],
        phase: str,
        save_path: Path,
        show: bool = False,
    ) -> None:
        """Plot confusion matrix for a specific phase if valid.

        Args:
            conf_matrix: Confusion matrix data
            phase: Phase name (train/val/test)
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if isinstance(conf_matrix, list) and isinstance(conf_matrix[0], list):
            # Handle history data
            if isinstance(conf_matrix[0][0], list):
                conf_matrix = conf_matrix[-1]

            conf_matrix = np.array(conf_matrix)

            if (
                len(conf_matrix.shape) == 2
                and conf_matrix.shape[0] == conf_matrix.shape[1]
            ):
                self.plot_confusion_matrix(
                    conf_matrix=conf_matrix,
                    classes=[str(i) for i in range(len(conf_matrix))],
                    save_path=save_path / f"confusion_matrix_{phase}.png",
                    show=show,
                )
