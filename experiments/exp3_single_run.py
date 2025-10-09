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
from data.data_loader import get_cifar10_special_pixel_dataloaders, get_cifar10_dataloaders


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


def compute_occlusion_sensitivity(model, trainloader, criterion, device, samples_per_class=100):
    """
    Compute occlusion sensitivity maps for CIFAR10 train set.

    For each class, takes the first samples_per_class images (deterministic across seeds)
    and computes occlusion sensitivity by measuring increase in loss when each pixel is occluded.
    Uses efficient batching: for each pixel position, creates batch of all images with that pixel occluded.
    Returns averaged occlusion map per class.

    Args:
        model: Trained model
        trainloader: Training data loader
        criterion: Loss function
        device: Device to compute on
        samples_per_class: Number of first samples per class to average over (default: 100)

    Returns:
        dict with:
            - 'occlusion_maps': np.array of shape (10, 32, 32) - averaged maps per class
            - 'sample_images': np.array of shape (10, 3, 32, 32) - first image per class
            - 'sample_labels': np.array of shape (10,) - labels for sample images
    """
    model.eval()

    # Collect first samples_per_class images per class (deterministic)
    class_samples = {i: [] for i in range(10)}
    class_labels = {i: [] for i in range(10)}

    with torch.no_grad():
        for inputs, labels in trainloader:
            for i in range(len(labels)):
                label = labels[i].item()
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(inputs[i])
                    class_labels[label].append(label)

            # Check if we have enough samples for all classes
            if all(len(class_samples[c]) >= samples_per_class for c in range(10)):
                break

    occlusion_maps = np.zeros((10, 32, 32))
    sample_images = np.zeros((10, 3, 32, 32))

    print("\nComputing occlusion sensitivity maps...")
    for class_idx in tqdm(range(10), desc="Classes", leave=False):
        # Stack all images for this class into a batch
        images_batch = torch.stack(class_samples[class_idx]).to(device)  # Shape: (100, 3, 32, 32)
        labels_batch = torch.tensor(class_labels[class_idx]).to(device)  # Shape: (100,)
        n_samples = images_batch.shape[0]

        # Get baseline losses for all images
        with torch.no_grad():
            outputs = model(images_batch)
            baseline_losses = torch.nn.functional.cross_entropy(outputs, labels_batch, reduction='none')
            baseline_losses = baseline_losses.cpu().numpy()  # Shape: (100,)

        # For each pixel position, occlude that pixel in all images and compute loss
        occlusion_map = np.zeros((32, 32))

        for pixel_idx in tqdm(range(32 * 32), desc=f"Class {class_idx}", leave=False):
            row = pixel_idx // 32
            col = pixel_idx % 32

            # Clone batch and occlude the pixel (set all 3 RGB channels to 0)
            occluded_batch = images_batch.clone()
            occluded_batch[:, :, row, col] = 0  # Occlude all 3 channels

            # Compute losses for occluded versions
            with torch.no_grad():
                outputs = model(occluded_batch)
                occluded_losses = torch.nn.functional.cross_entropy(outputs, labels_batch, reduction='none')
                occluded_losses = occluded_losses.cpu().numpy()  # Shape: (100,)

            # Average loss increase across all samples
            loss_increase = occluded_losses - baseline_losses
            occlusion_map[row, col] = np.mean(loss_increase)

        occlusion_maps[class_idx] = occlusion_map
        sample_images[class_idx] = class_samples[class_idx][0].cpu().numpy()

    return {
        'occlusion_maps': occlusion_maps,
        'sample_images': sample_images,
        'sample_labels': np.arange(10)
    }


def train_model(model, trainloader, testloader, device='cuda',
                lr=0.001, weight_decay=5e-4, max_epochs=200, target_train_acc=99.99,
                compute_occlusion=False):
    """
    Train model for max_epochs with AdamW optimizer (no early stopping, but tracks when target is reached).

    Args:
        model: The VGG model to train
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on (cuda or cpu)
        lr: Learning rate for AdamW
        weight_decay: Weight decay
        max_epochs: Maximum number of training epochs
        target_train_acc: Target train accuracy to track (default: 99.99%)
        compute_occlusion: If True, compute occlusion sensitivity at epoch 1 and final

    Returns:
        dict: Training metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    epochs_to_100pct = -1
    occlusion_epoch1 = None
    occlusion_final = None
    gap_epoch1 = None

    print(f"\nTraining for {max_epochs} epochs (tracking when {target_train_acc}% train accuracy is reached)...")
    pbar = tqdm(range(1, max_epochs + 1), desc="Training")

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
            print(f"\nEpoch {epoch}/{max_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Compute occlusion sensitivity at epoch 1
        if compute_occlusion and epoch == 1:
            print(f"\n{'='*80}")
            print(f"Computing occlusion sensitivity at epoch 1...")
            print(f"{'='*80}")
            occlusion_epoch1 = compute_occlusion_sensitivity(model, trainloader, criterion, device)
            gap_epoch1 = train_acc - test_acc
            print(f"Epoch 1 - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, Gap: {gap_epoch1:.2f}%")
            print(f"{'='*80}\n")

        # Track when we reach target train accuracy (but don't stop training)
        if train_acc >= target_train_acc and epochs_to_100pct == -1:
            epochs_to_100pct = epoch
            print(f"\nReached target train accuracy {target_train_acc}% at epoch {epoch}. Continuing training...")

    # Always use final epoch metrics (no early stopping)
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]

    # Compute occlusion sensitivity at final epoch
    if compute_occlusion:
        print(f"\n{'='*80}")
        print(f"Computing occlusion sensitivity at final epoch...")
        print(f"{'='*80}")
        occlusion_final = compute_occlusion_sensitivity(model, trainloader, criterion, device)
        print(f"{'='*80}\n")

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'final_test_loss': final_test_loss,
        'final_test_acc': final_test_acc,
        'generalization_gap': final_train_acc - final_test_acc,
        'total_epochs': len(train_accs),
        'epochs_to_100pct': epochs_to_100pct,
        'occlusion_epoch1': occlusion_epoch1,
        'occlusion_final': occlusion_final,
        'gap_epoch1': gap_epoch1
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
    parser.add_argument('--pixel_row', type=int, default=0,
                        help='Row for special pixel (0-31)')
    parser.add_argument('--pixel_col', type=int, default=0,
                        help='Column for special pixel (0-31)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--no_pixel', action='store_true',
                        help='Train without special pixel (use regular CIFAR10)')
    parser.add_argument('--no_bn', action='store_true',
                        help='Disable BatchNorm in variable VGG (default: BN enabled)')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Dropout probability in classifier')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--target_train_acc', type=float, default=99.99,
                        help='Target train accuracy to track')

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
    model = model_fn(num_classes=10, n_layers=args.n_layers, with_bn=(not args.no_bn), dropout_p=args.dropout_p)

    # Calculate noise level: force special pixel to be always correct (no noise)
    noise_level = 0.0 if not args.no_pixel else None

    pixel_mode = "No Pixel" if args.no_pixel else "With Special Pixel"
    print(f"\n{'='*80}")
    print(f"Experiment 3: {model_name} Variable Depth ({pixel_mode})")
    print(f"{'='*80}")
    print(f"Architecture:     {model_name}")
    print(f"Num Layers:       {args.n_layers}")
    if not args.no_pixel:
        print(f"Target MI:        {args.mi_bits} bits")
        print(f"Noise Level:      {noise_level:.4f}")
    else:
        print(f"Mode:             No special pixel")
    print(f"Seed:             {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Max Epochs:       {args.max_epochs}")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Load data
    if args.no_pixel:
        # Use regular CIFAR10 dataloaders (no special pixel)
        from data.data_loader import get_cifar10_transforms
        transform_train, transform_test = get_cifar10_transforms(augment=(not args.no_augment))

        import torchvision
        from torch.utils.data import DataLoader

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        testloader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    else:
        # Use special pixel dataloaders
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
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, target_train_acc=args.target_train_acc,
        compute_occlusion=True
    )

    # Prepare results
    is_valid = metrics['final_train_acc'] >= 99.99

    result = {
        'model': model_name,
        'n_layers': args.n_layers,
        'noise_level': noise_level,
        'mi_bits': args.mi_bits,
        'seed': args.seed,
        'valid': is_valid,
        'epochs_to_100pct': metrics['epochs_to_100pct'],
        'train_acc': metrics['final_train_acc'],
        'test_acc': metrics['final_test_acc'],
        'generalization_gap': metrics['generalization_gap'],
        'train_loss': metrics['final_train_loss'],
        'test_loss': metrics['final_test_loss'],
        'total_epochs': metrics['total_epochs'],
        'gap_epoch1': metrics['gap_epoch1'],
    }

    # Add occlusion data if available
    if metrics['occlusion_epoch1'] is not None:
        result.update({
            'occlusion_maps_epoch1': metrics['occlusion_epoch1']['occlusion_maps'],
            'sample_images_epoch1': metrics['occlusion_epoch1']['sample_images'],
            'sample_labels_epoch1': metrics['occlusion_epoch1']['sample_labels'],
            'occlusion_maps_final': metrics['occlusion_final']['occlusion_maps'],
            'sample_images_final': metrics['occlusion_final']['sample_images'],
            'sample_labels_final': metrics['occlusion_final']['sample_labels'],
        })

    if is_valid:
        print(f"\n{'='*80}")
        print(f"Training Complete (Valid - Reached 99.99%)")
        print(f"{'='*80}")
        print(f"Epochs to 100%:   {metrics['epochs_to_100pct']}")
        print(f"Train Accuracy:   {metrics['final_train_acc']:.2f}%")
        print(f"Test Accuracy:    {metrics['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Training Complete (Invalid - Did Not Reach 99.99%)")
        print(f"{'='*80}")
        print(f"Total Epochs:     {metrics['total_epochs']}")
        print(f"Best Train Acc:   {metrics['final_train_acc']:.2f}%")
        print(f"Test Acc at Best: {metrics['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming convention: vgg{X}var_layers{n}_seed{seed}_results.npz (or _nopixel_results.npz)
    arch_num = model_name.replace('VGG', '').lower()
    suffix = "_nopixel" if args.no_pixel else ""
    save_path = f"{args.output_dir}/vgg{arch_num}var_layers{args.n_layers}_seed{args.seed}{suffix}_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
