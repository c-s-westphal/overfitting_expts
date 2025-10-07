"""
Experiment 3: Variable-depth VGG models at fixed MI.

Studies how the number of convolutional layers affects generalization
within the same architecture family, using a unified classifier.
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

from models.vgg_variable_family import VGG11_Variable, VGG13_Variable, VGG16_Variable, VGG19_Variable
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


def get_lr_for_epoch(epoch):
    """Get learning rate for given epoch (1-indexed)."""
    if epoch <= 100:
        return 0.1
    elif epoch <= 150:
        return 0.01
    else:
        return 0.001


def train_model_adaptive(model, trainloader, testloader, device='cuda'):
    """
    Train model with adaptive duration (200-500 epochs).

    Trains for 200 epochs initially. If train_acc < 99%, extends to 500 epochs.
    Learning rate schedule based on current epoch: 0.1 (1-100), 0.01 (101-150), 0.001 (151+).

    Returns:
        dict: Training metrics including final accuracies and generalization gap
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # Initial training: 200 epochs
    initial_epochs = 200
    print(f"\nTraining for {initial_epochs} epochs...")
    pbar = tqdm(range(1, initial_epochs + 1), desc="Training")

    for epoch in pbar:
        # Update learning rate based on current epoch
        current_lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        pbar.set_postfix({
            'Epoch': epoch,
            'LR': current_lr,
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%'
        })

    # Check if we need to extend training
    final_train_acc = train_accs[-1]
    if final_train_acc < 99.0:
        max_epochs = 500
        print(f"\nTrain accuracy {final_train_acc:.2f}% < 99%. Extending to {max_epochs} epochs...")
        pbar = tqdm(range(initial_epochs + 1, max_epochs + 1), desc="Extended Training")

        for epoch in pbar:
            # Update learning rate based on current epoch
            current_lr = get_lr_for_epoch(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, testloader, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            pbar.set_postfix({
                'Epoch': epoch,
                'LR': current_lr,
                'Train Acc': f'{train_acc:.2f}%',
                'Test Acc': f'{test_acc:.2f}%'
            })

            # Early stop if we reach 99% during extended training
            if train_acc >= 99.0:
                print(f"\nReached 99% train accuracy at epoch {epoch}. Stopping training.")
                break

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
        'total_epochs': len(train_accs)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 3: Variable-depth VGG at fixed MI'
    )
    parser.add_argument('--arch', type=str, required=True,
                        choices=['vgg11var', 'vgg13var', 'vgg16var', 'vgg19var'],
                        help='VGG architecture variant')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of convolutional layers (VGG11: 4-8, VGG13: 4-10, VGG16: 4-13, VGG19: 4-16)')
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

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build model
    arch_map = {
        'vgg11var': ('VGG11', VGG11_Variable),
        'vgg13var': ('VGG13', VGG13_Variable),
        'vgg16var': ('VGG16', VGG16_Variable),
        'vgg19var': ('VGG19', VGG19_Variable),
    }

    model_name, model_fn = arch_map[args.arch]
    model = model_fn(num_classes=10, n_layers=args.n_layers)

    # Calculate noise level from target MI
    noise_level = noise_for_target_mi(args.mi_bits, num_classes=10)

    print(f"\n{'='*80}")
    print(f"Experiment 3: {model_name} Variable Depth")
    print(f"{'='*80}")
    print(f"Architecture:     {model_name}")
    print(f"Num Layers:       {args.n_layers}")
    print(f"Target MI:        {args.mi_bits} bits")
    print(f"Noise Level:      {noise_level:.4f}")
    print(f"Seed:             {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Load data
    trainloader, testloader = get_cifar10_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=noise_level,
        seed=args.seed
    )

    # Train model with adaptive duration
    metrics = train_model_adaptive(model, trainloader, testloader, device=args.device)

    # Prepare results
    is_valid = metrics['final_train_acc'] >= 99.0

    result = {
        'model': model_name,
        'n_layers': args.n_layers,
        'noise_level': noise_level,
        'mi_bits': args.mi_bits,
        'seed': args.seed,
        'valid': is_valid,
        'total_epochs': metrics['total_epochs']
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

    # Naming convention: vgg{X}var_layers{n}_seed{seed}_results.npz
    arch_num = model_name.replace('VGG', '').lower()
    save_path = f"{args.output_dir}/vgg{arch_num}var_layers{args.n_layers}_seed{args.seed}_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
