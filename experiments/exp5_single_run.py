"""
Experiment 5: Variable-depth All-Conv models on CIFAR-10 at fixed MI.

Studies how the number of convolutional layers affects generalization
within a simple all-convolutional architecture with fixed width.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conv_variable import AllConv_Variable
from data.data_loader import get_cifar10_special_pixel_dataloaders


def noise_for_target_mi(target_mi_bits, num_classes=10):
    """Convert target MI to noise level for special pixel."""
    max_mi = np.log2(num_classes)
    noise = (1.0 - (target_mi_bits / max_mi)) / 2.0
    return float(noise)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def train_model(model, trainloader, testloader, device='cuda',
                lr=0.001, weight_decay=1e-4, num_epochs=75):
    """
    Train model for fixed number of epochs with Adam optimizer.

    Args:
        model: The All-Conv model to train
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on (cuda or cpu)
        lr: Learning rate for Adam optimizer
        weight_decay: Weight decay (L2 regularization)
        num_epochs: Number of training epochs

    Returns:
        dict: Training metrics including final accuracies and generalization gap
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print(f"\nTraining for {num_epochs} epochs...")
    pbar = tqdm(range(1, num_epochs + 1), desc="Training")

    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        pbar.set_postfix({
            'Epoch': epoch,
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%'
        })

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_train_loss': train_losses[-1],
        'final_train_acc': final_train_acc,
        'final_test_loss': test_losses[-1],
        'final_test_acc': final_test_acc,
        'generalization_gap': final_train_acc - final_test_acc,
        'total_epochs': num_epochs
    }


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 5: Variable-depth All-Conv on CIFAR-10 at fixed MI'
    )
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of convolutional layers (1-5)')
    parser.add_argument('--mi_bits', type=float, default=2.5,
                        help='Target MI in bits for special pixel (default: 2.5)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--pixel_row', type=int, default=0,
                        help='Row for special pixel (0-31, default: 0)')
    parser.add_argument('--pixel_col', type=int, default=0,
                        help='Column for special pixel (0-31, default: 0)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=75,
                        help='Number of training epochs')
    parser.add_argument('--num_channels', type=int, default=128,
                        help='Number of channels in conv layers')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build model
    model = AllConv_Variable(num_classes=10, n_layers=args.n_layers, num_channels=args.num_channels)

    # Force special pixel to be always correct (no noise)
    noise_level = 0.0

    print(f"\n{'='*80}")
    print(f"Experiment 5: All-Conv Variable Depth on CIFAR-10")
    print(f"{'='*80}")
    print(f"Architecture:     All-Conv")
    print(f"Conv Layers:      {args.n_layers}")
    print(f"Num Channels:     {args.num_channels}")
    print(f"Total Params:     {model.count_parameters():,}")
    print(f"Target MI:        {args.mi_bits} bits")
    print(f"Noise Level:      {noise_level:.4f}")
    print(f"Seed:             {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Num Epochs:       {args.num_epochs}")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Load data
    trainloader, testloader = get_cifar10_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=noise_level,
        seed=args.seed,
        pixel_location=(args.pixel_row, args.pixel_col),
        augment=(not args.no_augment)
    )

    # Train model
    metrics = train_model(
        model, trainloader, testloader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay, num_epochs=args.num_epochs
    )

    # Prepare results
    is_valid = metrics['final_train_acc'] >= 99.0

    result = {
        'model': 'AllConv',
        'n_layers': args.n_layers,
        'num_channels': args.num_channels,
        'noise_level': noise_level,
        'mi_bits': args.mi_bits,
        'seed': args.seed,
        'valid': is_valid,
        'total_epochs': metrics['total_epochs'],
        'num_parameters': model.count_parameters()
    }

    if is_valid:
        result.update({
            'train_acc': metrics['final_train_acc'],
            'test_acc': metrics['final_test_acc'],
            'generalization_gap': metrics['generalization_gap'],
            'train_loss': metrics['final_train_loss'],
            'test_loss': metrics['final_test_loss'],
        })

        print(f"\n{'='*80}")
        print(f"Training Complete (Valid)")
        print(f"{'='*80}")
        print(f"Total Epochs:     {metrics['total_epochs']}")
        print(f"Train Accuracy:   {metrics['final_train_acc']:.2f}%")
        print(f"Test Accuracy:    {metrics['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Training Complete (Invalid - Train Acc < 99%)")
        print(f"{'='*80}")
        print(f"Total Epochs:     {metrics['total_epochs']}")
        print(f"Train Accuracy:   {metrics['final_train_acc']:.2f}%")
        print(f"{'='*80}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming convention: allconv_layers{n}_seed{seed}_results.npz
    save_path = f"{args.output_dir}/allconv_layers{args.n_layers}_seed{args.seed}_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
