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


def train_model_multi_epoch(model, trainloader, testloader, device='cuda', lr=0.001, weight_decay=5e-4,
                            max_epochs=500, target_train_acc=99.99, occlusion_interval=10):
    """
    Train model until target_train_acc is reached, with cosine annealing LR scheduler.
    Compute occlusion sensitivity every occlusion_interval epochs.

    Args:
        model: The VGG model to train
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on (cuda or cpu)
        lr: Initial learning rate for AdamW
        weight_decay: Weight decay
        max_epochs: Maximum number of training epochs
        target_train_acc: Target training accuracy to stop (default: 99.99%)
        occlusion_interval: Compute occlusion every N epochs (default: 10)

    Returns:
        dict: Training metrics including occlusion at intervals
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    print(f"\nTraining until {target_train_acc}% train accuracy (max {max_epochs} epochs)...")
    print(f"Using cosine annealing LR scheduler")
    print(f"Computing occlusion every {occlusion_interval} epochs")

    # Store metrics for all epochs
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    generalization_gaps = []
    occlusion_data = {}

    final_epoch = max_epochs
    pbar = tqdm(range(1, max_epochs + 1), desc="Training")

    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        gap = train_acc - test_acc

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        generalization_gaps.append(gap)

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Epoch': epoch,
            'Train': f'{train_acc:.2f}%',
            'Test': f'{test_acc:.2f}%',
            'LR': f'{current_lr:.6f}'
        })

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | LR: {current_lr:.6f}")

        # Compute occlusion sensitivity at intervals (epoch 1 and every occlusion_interval epochs)
        if epoch == 1 or epoch % occlusion_interval == 0:
            print(f"\n{'='*80}")
            print(f"Computing occlusion sensitivity at epoch {epoch}...")
            print(f"{'='*80}")
            occlusion = compute_occlusion_sensitivity(model, trainloader, criterion, device)
            occlusion_data[epoch] = {
                'maps': occlusion['occlusion_maps'],
                'images': occlusion['sample_images'],
                'labels': occlusion['sample_labels'],
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap
            }
            print(f"Epoch {epoch} - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, Gap: {gap:.2f}%")
            print(f"{'='*80}\n")

        # Stop when we reach target train accuracy
        if train_acc >= target_train_acc:
            print(f"\nReached target train accuracy {target_train_acc}% at epoch {epoch}. Stopping.")
            final_epoch = epoch
            # Compute final occlusion if not already computed this epoch
            if epoch not in occlusion_data:
                print(f"\n{'='*80}")
                print(f"Computing occlusion sensitivity at final epoch {epoch}...")
                print(f"{'='*80}")
                occlusion = compute_occlusion_sensitivity(model, trainloader, criterion, device)
                occlusion_data[epoch] = {
                    'maps': occlusion['occlusion_maps'],
                    'images': occlusion['sample_images'],
                    'labels': occlusion['sample_labels'],
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'gap': gap
                }
                print(f"{'='*80}\n")
            break

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
        'total_epochs': final_epoch,
        'occlusion_data': occlusion_data,
    }

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
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum number of training epochs')
    parser.add_argument('--target_train_acc', type=float, default=99.99,
                        help='Target train accuracy to stop training')
    parser.add_argument('--occlusion_interval', type=int, default=10,
                        help='Compute occlusion maps every N epochs')

    args = parser.parse_args()

    # Check device availability and handle CUDA errors
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            try:
                # Test CUDA initialization
                torch.cuda.current_device()
                print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            except RuntimeError as e:
                print(f"WARNING: CUDA initialization failed with error: {e}")
                print(f"Falling back to CPU.")
                args.device = 'cpu'

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
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
    print(f"Experiment 6: {model_name} Full Depth - Occlusion (until {args.target_train_acc}%)")
    print(f"{'='*80}")
    print(f"Architecture:     {model_name}")
    print(f"Num Layers:       {full_depth} (full depth)")
    print(f"Special Pixel:    {pixel_location} (center)")
    print(f"Noise Level:      {noise_level:.4f} (always correct)")
    print(f"Seed:             {args.seed}")
    print(f"Max Epochs:       {args.max_epochs}")
    print(f"Target Acc:       {args.target_train_acc}%")
    print(f"Occlusion Interval: Every {args.occlusion_interval} epochs")
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

    # Train model until target accuracy
    metrics = train_model_multi_epoch(
        model, trainloader, testloader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, target_train_acc=args.target_train_acc,
        occlusion_interval=args.occlusion_interval
    )

    # Prepare results
    result = {
        'model': model_name,
        'n_layers': full_depth,
        'noise_level': noise_level,
        'pixel_location': pixel_location,
        'seed': args.seed,
        'total_epochs': metrics['total_epochs'],
        'train_accs': np.array(metrics['train_accs']),
        'test_accs': np.array(metrics['test_accs']),
        'generalization_gaps': np.array(metrics['generalization_gaps']),
        'train_losses': np.array(metrics['train_losses']),
        'test_losses': np.array(metrics['test_losses']),
        'final_train_acc': metrics['final_train_acc'],
        'final_test_acc': metrics['final_test_acc'],
        'final_generalization_gap': metrics['final_generalization_gap'],
    }

    # Add occlusion data for each epoch it was computed
    occlusion_epochs = sorted(metrics['occlusion_data'].keys())
    result['occlusion_epochs'] = np.array(occlusion_epochs)
    for epoch in occlusion_epochs:
        occ = metrics['occlusion_data'][epoch]
        result[f'occlusion_maps_epoch{epoch}'] = occ['maps']
        result[f'sample_images_epoch{epoch}'] = occ['images']
        result[f'occlusion_train_acc_epoch{epoch}'] = occ['train_acc']
        result[f'occlusion_test_acc_epoch{epoch}'] = occ['test_acc']
        result[f'occlusion_gap_epoch{epoch}'] = occ['gap']

    is_valid = metrics['final_train_acc'] >= args.target_train_acc

    print(f"\n{'='*80}")
    if is_valid:
        print(f"Training Complete (Valid - Reached {args.target_train_acc}%)")
    else:
        print(f"Training Complete (Invalid - Did Not Reach {args.target_train_acc}%)")
    print(f"{'='*80}")
    print(f"Total Epochs:           {metrics['total_epochs']}")
    print(f"Final Train Accuracy:   {metrics['final_train_acc']:.2f}%")
    print(f"Final Test Accuracy:    {metrics['final_test_acc']:.2f}%")
    print(f"Final Gen. Gap:         {metrics['final_generalization_gap']:.2f}%")
    print(f"Occlusion epochs:       {occlusion_epochs}")
    print(f"{'='*80}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming convention: vgg{X}_seed{seed}_results.npz
    arch_num = model_name.lower()
    save_path = f"{args.output_dir}/{arch_num}_seed{args.seed}_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")


if __name__ == "__main__":
    main()
