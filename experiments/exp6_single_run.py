"""
Experiment 6: Full-depth VGG models with occlusion sensitivity tracking.

Studies occlusion sensitivity across training epochs for different VGG architecture sizes
(VGG9, VGG11, VGG13, VGG16, VGG19) using full depth for each architecture.
Special pixel at center (16, 16) with 0.0 noise.
Can train for 1 or multiple epochs (use --epochs flag).
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

from models.vgg_variable_family import VGG9_Variable, VGG11_Variable, VGG13_Variable, VGG16_Variable, VGG19_Variable
from data.data_loader import get_cifar10_special_pixel_dataloaders


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


def train_model_multi_epoch(model, trainloader, testloader, device='cuda', lr=0.001, weight_decay=5e-4, epochs=1):
    """
    Train model for multiple epochs and compute occlusion sensitivity at each epoch.

    Args:
        model: The VGG model to train
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on (cuda or cpu)
        lr: Learning rate for AdamW
        weight_decay: Weight decay
        epochs: Number of epochs to train (default: 1)

    Returns:
        dict: Training metrics including occlusion at each epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nTraining for {epochs} epoch(s)...")

    # Store metrics for all epochs
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    generalization_gaps = []
    occlusion_data = {}

    # Train for specified number of epochs
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        gap = train_acc - test_acc

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        generalization_gaps.append(gap)

        print(f"\nEpoch {epoch} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Compute occlusion sensitivity at this epoch
        print(f"\n{'='*80}")
        print(f"Computing occlusion sensitivity at epoch {epoch}...")
        print(f"{'='*80}")
        occlusion = compute_occlusion_sensitivity(model, trainloader, criterion, device)
        occlusion_data[f'occlusion_epoch{epoch}'] = occlusion
        print(f"Epoch {epoch} - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, Gap: {gap:.2f}%")
        print(f"{'='*80}\n")

    # Return comprehensive results
    result = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'final_train_loss': train_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_test_loss': test_losses[-1],
        'final_test_acc': test_accs[-1],
        'final_generalization_gap': generalization_gaps[-1],
        'epochs': epochs,
    }

    # Add all occlusion data
    result.update(occlusion_data)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 6: Full-depth VGG with epoch 1 occlusion'
    )
    parser.add_argument('--arch', type=str, required=True,
                        choices=['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='VGG architecture variant')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train (default: 1)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build model - use full depth for each architecture
    arch_map = {
        'vgg9': ('VGG9', VGG9_Variable, 6),
        'vgg11': ('VGG11', VGG11_Variable, 8),
        'vgg13': ('VGG13', VGG13_Variable, 10),
        'vgg16': ('VGG16', VGG16_Variable, 13),
        'vgg19': ('VGG19', VGG19_Variable, 16),
    }

    model_name, model_fn, full_depth = arch_map[args.arch]
    model = model_fn(num_classes=10, n_layers=full_depth, with_bn=True, dropout_p=0.0)

    # Special pixel at center (16, 16) with no noise (always correct)
    noise_level = 0.0
    pixel_location = (16, 16)

    print(f"\n{'='*80}")
    print(f"Experiment 6: {model_name} Full Depth - Occlusion ({args.epochs} epoch(s))")
    print(f"{'='*80}")
    print(f"Architecture:     {model_name}")
    print(f"Num Layers:       {full_depth} (full depth)")
    print(f"Special Pixel:    {pixel_location} (center)")
    print(f"Noise Level:      {noise_level:.4f} (always correct)")
    print(f"Seed:             {args.seed}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Load data with special pixel
    trainloader, testloader = get_cifar10_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=noise_level,
        seed=args.seed,
        pixel_location=pixel_location,
        augment=(not args.no_augment)
    )

    # Train model for specified epochs
    metrics = train_model_multi_epoch(
        model, trainloader, testloader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
    )

    # Prepare results
    result = {
        'model': model_name,
        'n_layers': full_depth,
        'noise_level': noise_level,
        'pixel_location': pixel_location,
        'seed': args.seed,
        'epochs': args.epochs,
        'train_accs': metrics['train_accs'],
        'test_accs': metrics['test_accs'],
        'generalization_gaps': metrics['generalization_gaps'],
        'train_losses': metrics['train_losses'],
        'test_losses': metrics['test_losses'],
        'final_train_acc': metrics['final_train_acc'],
        'final_test_acc': metrics['final_test_acc'],
        'final_generalization_gap': metrics['final_generalization_gap'],
    }

    # Add all occlusion data for each epoch
    for epoch in range(1, args.epochs + 1):
        occlusion_key = f'occlusion_epoch{epoch}'
        if occlusion_key in metrics:
            result[f'occlusion_maps_epoch{epoch}'] = metrics[occlusion_key]['occlusion_maps']
            result[f'sample_images_epoch{epoch}'] = metrics[occlusion_key]['sample_images']
            result[f'sample_labels_epoch{epoch}'] = metrics[occlusion_key]['sample_labels']

    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"{'='*80}")
    print(f"Final Train Accuracy:   {metrics['final_train_acc']:.2f}%")
    print(f"Final Test Accuracy:    {metrics['final_test_acc']:.2f}%")
    print(f"Final Gen. Gap:         {metrics['final_generalization_gap']:.2f}%")
    print(f"{'='*80}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming convention: vgg{X}_seed{seed}_results.npz or vgg{X}_seed{seed}_{N}epochs_results.npz
    arch_num = model_name.lower()
    if args.epochs == 1:
        save_path = f"{args.output_dir}/{arch_num}_seed{args.seed}_results.npz"
    else:
        save_path = f"{args.output_dir}/{arch_num}_seed{args.seed}_{args.epochs}epochs_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
