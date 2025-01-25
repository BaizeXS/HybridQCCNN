"""Main program entry, supporting model training, testing, prediction, and visualization.

Supported features:
1. Train model
2. Test model performance
3. Use model to predict
4. Visualize training metrics
5. Compare training metrics of multiple models
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from config import ConfigManager
from utils.data_management import DatasetManager
from utils.model_management import ModelManager
from utils.visualization import MetricsPlotter

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MainProgram:
    """Main program class, encapsulating various functionalities"""

    def __init__(self):
        """Initialize the main program.

        Initialize various managers, but the specific dataset and model managers
        will be set when used based on the configuration.
        """
        self.config_manager = ConfigManager()
        self.metrics_plotter = MetricsPlotter()
        self.current_config = None
        self.dataset_manager = None
        self.model_manager = None

    def _setup_managers(self, config_path: Union[str, Path]):
        """Set up managers based on the configuration file.

        Args:
            config_path: The path to the configuration file.
        """
        # Load new configuration
        self.current_config = self.config_manager.load_config(config_path)

        # Set up dataset manager
        data_dir = Path(self.current_config.data.dataset_path or "./datasets")
        self.dataset_manager = DatasetManager(
            config=self.current_config.data,
            data_dir=data_dir,
        )

        # Set up model manager
        self.model_manager = ModelManager(
            config=self.current_config,
            model_name=self.current_config.model.name,
        )

    def train(
        self,
        config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """Train the model.

        Args:
            config_path: The path to the configuration file.
            checkpoint_path: The path to the checkpoint file, if provided,
                training will continue from the checkpoint.
        """
        # Set up managers
        self._setup_managers(config_path)

        # Get data loaders
        if self.dataset_manager is None:
            raise ValueError(
                "Dataset manager not initialized. Please call _setup_managers first."
            )
        train_loader, val_loader, test_loader = self.dataset_manager.get_data_loaders()

        # Check if model manager is initialized
        if self.model_manager is None:
            raise ValueError(
                "Model manager not initialized. Please call _setup_managers first."
            )
        # If a checkpoint is provided, load it and continue training
        start_epoch = 0
        if checkpoint_path:
            start_epoch = self.model_manager.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")

        # Train the model
        # TODO: DO NOT SUPPORT TRAINING FROM CHECKPOINT
        self.model_manager.train(train_loader, val_loader)

        # Test the model
        test_metrics = self.model_manager.test(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

    def test(
        self,
        config_path: Union[str, Path],
        checkpoint_path_or_weights_path: Union[str, Path],
        is_checkpoint: bool = False,
    ):
        """Test the model.

        Args:
            config_path: The path to the configuration file.
            checkpoint_path_or_weights_path: The path to the checkpoint file or weights file.
            is_checkpoint: Whether the model file is a checkpoint file.
        """
        # Set up managers
        self._setup_managers(config_path)

        # Get test data loader
        if self.dataset_manager is None:
            raise ValueError(
                "Dataset manager not initialized. Please call _setup_managers first."
            )
        _, _, test_loader = self.dataset_manager.get_data_loaders()

        # Check if model manager is initialized
        if self.model_manager is None:
            raise ValueError(
                "Model manager not initialized. Please call _setup_managers first."
            )

        # Load model from checkpoint or weights
        if is_checkpoint:
            self.model_manager.load_checkpoint(checkpoint_path_or_weights_path)
        else:
            self.model_manager.load_weights(checkpoint_path_or_weights_path)

        # Test the model
        test_metrics = self.model_manager.test(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

    def predict(
        self,
        config_path: Union[str, Path],
        checkpoint_path_or_weights_path: Union[str, Path],
        image_path: Union[str, Path],
        is_checkpoint: bool = False,
    ):
        """Use the model to predict.

        Args:
            config_path: The path to the configuration file.
            checkpoint_path_or_weights_path: The path to the checkpoint file or weights file.
            image_path: The path to the image to predict.
            is_checkpoint: Whether the model file is a checkpoint file.
        """
        # Set up managers
        self._setup_managers(config_path)

        # Check if model manager is initialized
        if self.model_manager is None:
            raise ValueError(
                "Model manager not initialized. Please call _setup_managers first."
            )

        # Load model from checkpoint or weights
        if is_checkpoint:
            self.model_manager.load_checkpoint(checkpoint_path_or_weights_path)
        else:
            self.model_manager.load_weights(checkpoint_path_or_weights_path)

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.tensor(self._get_transform()(image)).unsqueeze(0)
        image_tensor = image_tensor.to(self.model_manager.device)

        # Predict
        with torch.no_grad():
            outputs = self.model_manager.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = int(torch.argmax(probs, dim=1)[0])
            confidence = float(probs[0][pred_class])

        # Log results
        logger.info(f"Image: {image_path}")
        logger.info(f"Class: {pred_class}, Confidence: {confidence:.4f}")
        return pred_class, confidence

    def _get_transform(self) -> transforms.Compose:
        """Get image transform from config or use default."""
        # Check if current config is initialized
        if self.current_config is None:
            raise ValueError(
                "Current config not initialized. Please call _setup_managers first."
            )

        # Create transform from config
        transform_list = []
        if self.current_config.data.test_transforms:
            for t in self.current_config.data.test_transforms:
                transform_class = getattr(transforms, t["name"])
                transform_list.append(transform_class(**t.get("args", {})))

        # If no transform is specified or test_transforms is None, use default transform
        if not transform_list:
            transform_list = [
                transforms.Resize(self.current_config.data.input_shape[1:]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]

        return transforms.Compose(transform_list)

    def visualize_metrics(
        self,
        metrics_path: Union[str, Path],
        metric_names: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """Visualize training metrics.

        Args:
            metrics_path: The path to the metrics file.
            metric_names: The names of the metrics to visualize, if None,
                all metrics are visualized.
            phases: The phases to visualize, if None, all phases are visualized.
            save_dir: The directory to save the plots.
            show: Whether to display the plots.
        """
        self.metrics_plotter.plot_from_saved_metrics(
            metrics_path=metrics_path,
            metric_names=metric_names or [],
            phases=phases or [],
            save_dir=save_dir,
            show=show,
        )

    def compare_metrics(
        self,
        metrics_paths: List[Union[str, Path]],
        metric_names: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """Compare training metrics of multiple models.

        Args:
            metrics_paths: The paths to the metrics files.
            metric_names: The names of the metrics to compare, if None,
                all metrics are compared.
            model_names: The names of the models to compare, if None,
                all models are compared.
            phases: The phases to compare, if None, all phases are compared.
            save_dir: The directory to save the plots.
            show: Whether to display the plots.
        """
        # Load all metrics data
        all_metrics = {}
        for path in metrics_paths:
            path = Path(path)
            model_name = path.stem
            if model_names and model_name not in model_names:
                continue
            with open(path) as f:
                all_metrics[model_name] = f.read()

        # Plot comparison
        self.metrics_plotter.plot_metrics_comparison(
            data=all_metrics,
            model_names=model_names or [],
            metric_names=metric_names or [],
            phases=phases or [],
            save_dir=save_dir,
            show=show,
        )


def main():
    """Main function, parse command line arguments and execute corresponding functionality"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum Convolutional Neural Network Model Training and Usage Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path"
    )
    train_parser.add_argument(
        "--checkpoint", help="Checkpoint file path, for continuing training"
    )

    # test command
    test_parser = subparsers.add_parser("test", help="Test model performance")
    test_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path"
    )
    test_parser.add_argument(
        "-w", "--weights", required=True, help="Model weights or checkpoint file path"
    )
    test_parser.add_argument(
        "--is-checkpoint",
        action="store_true",
        help="Whether the weights file is a checkpoint file",
    )

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Use model to predict")
    predict_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path"
    )
    predict_parser.add_argument(
        "-w", "--weights", required=True, help="Model weights or checkpoint file path"
    )
    predict_parser.add_argument(
        "-i", "--image", required=True, help="Image path to predict"
    )
    predict_parser.add_argument(
        "--is-checkpoint",
        action="store_true",
        help="Whether the weights file is a checkpoint file",
    )

    # viz-metrics command
    viz_metrics_parser = subparsers.add_parser(
        "viz-metrics", help="Visualize training metrics"
    )
    viz_metrics_parser.add_argument(
        "-f", "--file", required=True, help="Metrics file path"
    )
    viz_metrics_parser.add_argument(
        "--metric-names", nargs="+", help="The names of the metrics to visualize"
    )
    viz_metrics_parser.add_argument(
        "--phases", nargs="+", help="The phases to visualize"
    )
    viz_metrics_parser.add_argument("-o", "--output-dir", help="Save directory")
    viz_metrics_parser.add_argument(
        "--no-show", action="store_true", help="Do not display images"
    )

    # compare-metrics command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare training metrics of multiple models"
    )
    compare_parser.add_argument(
        "-f", "--files", nargs="+", required=True, help="Metrics file paths"
    )
    compare_parser.add_argument(
        "--metric-names", nargs="+", help="The names of the metrics to compare"
    )
    compare_parser.add_argument(
        "--model-names", nargs="+", help="The names of the models to compare"
    )
    compare_parser.add_argument("--phases", nargs="+", help="The phases to compare")
    compare_parser.add_argument("-o", "--output-dir", help="Save directory")
    compare_parser.add_argument(
        "--no-show", action="store_true", help="Do not display images"
    )

    try:
        # Parse arguments
        args = parser.parse_args()

        # Create program instance
        program = MainProgram()

        # Execute corresponding command
        if args.command == "train":
            program.train(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
            )
        elif args.command == "test":
            program.test(
                config_path=args.config,
                checkpoint_path_or_weights_path=args.weights,
                is_checkpoint=args.is_checkpoint,
            )
        elif args.command == "predict":
            program.predict(
                config_path=args.config,
                checkpoint_path_or_weights_path=args.weights,
                image_path=args.image,
                is_checkpoint=args.is_checkpoint,
            )
        elif args.command == "viz-metrics":
            program.visualize_metrics(
                metrics_path=args.file,
                metric_names=args.metric_names,
                phases=args.phases,
                save_dir=args.output_dir,
                show=not args.no_show,
            )
        elif args.command == "compare":
            program.compare_metrics(
                metrics_paths=args.files,
                metric_names=args.metric_names,
                model_names=args.model_names,
                phases=args.phases,
                save_dir=args.output_dir,
                show=not args.no_show,
            )

    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
