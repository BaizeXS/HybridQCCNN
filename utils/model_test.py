import argparse
import json
import logging
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, confusion_matrix


from utils.model_utils import set_random_seed, get_models, get_transforms, get_dataset
from utils.visualization import save_confusion_matrix


def test(model_name, model, test_loader, criterion, device, output_dir):
    logger = logging.getLogger(__name__)
    logger.info(f"Start testing.")

    # Move the model to the specified device
    model = model.to(device)

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_samples = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Testing")
        for img, label in test_bar:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)

            # test_loss += abs(loss.item()) * img.size(0)
            test_loss += loss.item() * img.size(0)
            accuracy = torch.sum(torch.argmax(output, dim=1) == label).item()
            test_acc += accuracy
            test_samples += img.size(0)

            test_labels.extend(label.cpu().numpy())
            test_predictions.extend(torch.argmax(output, dim=1).cpu().numpy())

            test_bar.set_postfix(loss=loss.item(), accuracy=accuracy / img.size(0))

    test_loss /= test_samples
    test_acc /= test_samples
    test_recall = recall_score(test_labels, test_predictions, average='macro')
    test_f1 = f1_score(test_labels, test_predictions, average='macro')
    confusion_matrix_data = confusion_matrix(test_labels, test_predictions)

    # Save evaluation metrics
    output_path = str(os.path.join(output_dir, model_name))
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_recall": test_recall,
        "test_f1": test_f1,
    }
    with open(os.path.join(output_path, f"{model_name}_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Confusion Matrix
    save_confusion_matrix(confusion_matrix_data, os.path.join(output_path, f"{model_name}_confusion_matrix.png"))

    logger.info(f"Testing completed. Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum-Classical Model Testing")
    parser.add_argument("--model", type=str, default="ClassicNet", help="Model name (default: ClassicNet)")
    parser.add_argument("--dataset", type=str, default="FashionMNIST", help="Dataset name (default: FashionMNIST)")
    parser.add_argument("--data-dir", type=str, default="../datasets", help="Dataset directory (default: ../datasets)")
    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="../output", help="Output directory (default: ../output)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (default: None, auto-detect)")
    parser.add_argument("--qdevice", type=str, default="default.qubit", help="Quantum device (default: default.qubit)")
    parser.add_argument("--qdevice-kwargs", type=json.loads, default=None, help="Quantum device kwargs (default: None)")
    parser.add_argument("--diff-method", type=str, default="best", help="Differentiation method (default: best)")

    args = parser.parse_args()
    # qdevice_kwargs = json.loads(args.qdevice_kwargs) if args.qdevice_kwargs else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Set Random Seed
    set_random_seed(args.seed)

    # Set Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set Model
    models = get_models(args.dataset, args.qdevice, args.qdevice_kwargs, args.diff_method)
    model = models[args.model]
    model_weights_path = str(os.path.join(args.output_dir, args.model, f"{args.model}_model.pth"))
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # Set Criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Set Datasets
    train_transform, test_transform = get_transforms(args.model, args.dataset)
    _, test_loader = get_dataset(args.dataset, args.data_dir, train_transform, test_transform, args.batch_size)

    # Set Output Dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Test
    test(args.model, model, test_loader, criterion, device, args.output_dir)


if __name__ == "__main__":
    main()
