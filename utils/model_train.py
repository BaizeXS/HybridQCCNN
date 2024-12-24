import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model_utils import save_evaluation_metrics, set_random_seed, get_models, get_transforms, get_dataset


def train(model_name, model, train_loader, test_loader, optimizer, scheduler, criterion, device, num_epochs, output_dir,
          tensorboard_dir, aux_weight=0.4):
    logger = logging.getLogger(__name__)
    logger.info(f"Start training for {num_epochs} epochs.")

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_weights_path = os.path.join(str(model_dir), f"{model_name}_model.pth")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Move the model to the specified device
    model = model.to(device)

    # Initialize lists to store statistics
    best_test_acc = 0.0
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        for img, label in train_bar:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()

            if 'GoogLeNet' in model_name:
                output, aux_output = model(img)
                loss1 = criterion(output, label)
                loss2 = criterion(aux_output, label)
                loss = loss1 + aux_weight * loss2
            else:
                output = model(img)
                loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += abs(loss.item()) * img.size(0)
            accuracy = torch.sum(torch.argmax(output, dim=1) == label).item()
            train_acc += accuracy
            train_samples += img.size(0)

            train_bar.set_postfix(loss=loss.item(), accuracy=accuracy / img.size(0))

        train_loss /= train_samples
        train_acc /= train_samples
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Testing
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch} - Testing")
            for img, label in test_bar:
                img, label = img.to(device), label.to(device)
                output = model(img)
                loss = criterion(output, label)

                test_loss += abs(loss.item()) * img.size(0)
                accuracy = torch.sum(torch.argmax(output, dim=1) == label).item()
                test_acc += accuracy
                test_samples += img.size(0)

                test_bar.set_postfix(loss=loss.item(), accuracy=accuracy / img.size(0))

        test_loss /= test_samples
        test_acc /= test_samples
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_weights_path)

        elapsed_time = time.time() - start_time
        logger.info(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Elapsed Time: {elapsed_time:.2f}s")

    writer.close()

    # Save evaluation metrics
    save_evaluation_metrics(model_name, train_loss_history, "train_loss", output_dir)
    save_evaluation_metrics(model_name, train_acc_history, "train_accuracy", output_dir)
    save_evaluation_metrics(model_name, test_loss_history, "test_loss", output_dir)
    save_evaluation_metrics(model_name, test_acc_history, "test_accuracy", output_dir)
    logger.info(f"Training completed. Best test accuracy: {best_test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Quantum-Classical Model Training")
    parser.add_argument("--model", type=str, default="ClassicNet", help="Model name (default: ClassicNet)")
    parser.add_argument("--dataset", type=str, default="FashionMNIST", help="Dataset name (default: FashionMNIST)")
    parser.add_argument("--data-dir", type=str, default="../datasets", help="Dataset directory (default: ../datasets)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (default: 10)")
    parser.add_argument("--output-dir", type=str, default="../output", help="Output directory (default: ../output)")
    parser.add_argument("--tensorboard-dir", type=str, default="../tensorboard",
                        help="Tensorboard directory (default: ../tensorboard)")
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
    model.to(device)

    # Set Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Set Datasets
    train_transform, test_transform = get_transforms(args.model, args.dataset)
    train_loader, test_loader = get_dataset(args.dataset, args.data_dir, train_transform, test_transform,
                                            args.batch_size)

    # Set Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Train
    train(args.model, model, train_loader, test_loader, optimizer, scheduler, criterion, device, args.epochs,
          args.output_dir, args.tensorboard_dir)


if __name__ == "__main__":
    main()
