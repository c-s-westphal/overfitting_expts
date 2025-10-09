"""
Experiment 4: Variable-depth MLP models on MNIST at fixed MI.

Studies how the number of hidden layers affects generalization
within the same architecture family (fully-connected MLPs).
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

from models.mlp_variable import MLP_Variable
from data.data_loader import get_mnist_special_pixel_dataloaders, get_mnist_dataloaders


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


def compute_occlusion_sensitivity(model, trainloader, criterion, device, samples_per_class=256):
    """
    Compute occlusion sensitivity maps for train set.

    For each class, takes the first samples_per_class images (deterministic across seeds)
    and computes occlusion sensitivity by measuring increase in loss when each pixel is occluded.
    Uses efficient batching: for each pixel position, creates batch of all images with that pixel occluded.
    Returns averaged occlusion map per class.

    Args:
        model: Trained model
        trainloader: Training data loader
        criterion: Loss function
        device: Device to compute on
        samples_per_class: Number of first samples per class to average over (default: 256)

    Returns:
        dict with:
            - 'occlusion_maps': np.array of shape (10, 28, 28) - averaged maps per class
            - 'sample_images': np.array of shape (10, 28, 28) - first image per class
            - 'sample_labels': np.array of shape (10,) - class labels (0-9)
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

            # Check if we have enough samples
            if all(len(class_samples[c]) >= samples_per_class for c in range(10)):
                break

    # Compute occlusion maps for each class
    occlusion_maps = np.zeros((10, 28, 28))
    sample_images = np.zeros((10, 28, 28))

    print("\nComputing occlusion sensitivity maps...")
    for class_idx in tqdm(range(10), desc="Classes", leave=False):
        # Stack all images for this class into a batch
        images_batch = torch.stack(class_samples[class_idx]).to(device)  # Shape: (256, 1, 28, 28)
        labels_batch = torch.tensor(class_labels[class_idx]).to(device)  # Shape: (256,)
        n_samples = images_batch.shape[0]

        # Get baseline losses for all images
        with torch.no_grad():
            outputs = model(images_batch)
            baseline_losses = torch.nn.functional.cross_entropy(outputs, labels_batch, reduction='none')
            baseline_losses = baseline_losses.cpu().numpy()  # Shape: (256,)

        # For each pixel position, occlude that pixel in all images and compute loss
        occlusion_map = np.zeros((28, 28))

        for pixel_idx in range(784):
            row = pixel_idx // 28
            col = pixel_idx % 28

            # Clone batch and occlude this pixel in all images
            occluded_batch = images_batch.clone()
            occluded_batch[:, 0, row, col] = 0

            # Compute losses for occluded versions
            with torch.no_grad():
                outputs = model(occluded_batch)
                occluded_losses = torch.nn.functional.cross_entropy(outputs, labels_batch, reduction='none')
                occluded_losses = occluded_losses.cpu().numpy()  # Shape: (256,)

            # Average loss increase across all samples
            loss_increase = occluded_losses - baseline_losses
            occlusion_map[row, col] = np.mean(loss_increase)

        occlusion_maps[class_idx] = occlusion_map

        # Store first image as sample
        sample_images[class_idx] = class_samples[class_idx][0].cpu().numpy().squeeze()

    print("Occlusion sensitivity computation complete.\n")

    return {
        'occlusion_maps': occlusion_maps,
        'sample_images': sample_images,
        'sample_labels': np.arange(10)
    }


def train_model(model, trainloader, testloader, device='cuda',
                lr=0.001, weight_decay=1e-4, max_epochs=200, target_train_acc=99.0,
                compute_occlusion=False):
    """
    Train model until reaching target train accuracy or max epochs with Adam optimizer.

    Args:
        model: The MLP model to train
        trainloader: Training data loader
        testloader: Test data loader
        device: Device to train on (cuda or cpu)
        lr: Learning rate for Adam optimizer
        weight_decay: Weight decay (L2 regularization)
        max_epochs: Maximum number of training epochs
        target_train_acc: Target training accuracy to stop (default: 99.0%)
        compute_occlusion: If True, compute occlusion sensitivity at epoch 2 and final

    Returns:
        dict: Training metrics including epochs_to_99pct, accuracies, and generalization gap
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    epochs_to_99pct = -1
    best_train_acc = 0.0
    best_train_acc_epoch = -1

    # Storage for occlusion sensitivity
    occlusion_epoch2 = None
    occlusion_final = None
    gap_epoch2 = None

    print(f"\nTraining until {target_train_acc}% train accuracy or {max_epochs} epochs...")
    pbar = tqdm(range(1, max_epochs + 1), desc="Training")

    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Track best train accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_acc_epoch = epoch - 1  # 0-indexed for list access

        pbar.set_postfix({
            'Epoch': epoch,
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%'
        })

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{max_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Compute occlusion sensitivity at epoch 2
        if compute_occlusion and epoch == 2:
            print(f"\n{'='*80}")
            print(f"Computing occlusion sensitivity at epoch 2...")
            print(f"{'='*80}")
            occlusion_epoch2 = compute_occlusion_sensitivity(model, trainloader, criterion, device)
            gap_epoch2 = train_acc - test_acc
            print(f"Epoch 2 - Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, Gap: {gap_epoch2:.2f}%")
            print(f"{'='*80}\n")

        # Check if we've reached target train accuracy
        if train_acc >= target_train_acc:
            epochs_to_99pct = epoch
            print(f"\nReached target train accuracy {target_train_acc}% at epoch {epoch}. Stopping training.")
            break

    # Determine final metrics
    if epochs_to_99pct != -1:
        # We reached target, use final epoch metrics
        final_train_acc = train_accs[-1]
        final_test_acc = test_accs[-1]
        final_train_loss = train_losses[-1]
        final_test_loss = test_losses[-1]
    else:
        # Never reached target, use best train_acc epoch
        final_train_acc = train_accs[best_train_acc_epoch]
        final_test_acc = test_accs[best_train_acc_epoch]
        final_train_loss = train_losses[best_train_acc_epoch]
        final_test_loss = test_losses[best_train_acc_epoch]

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
        'epochs_to_99pct': epochs_to_99pct,
        'occlusion_epoch2': occlusion_epoch2,
        'occlusion_final': occlusion_final,
        'gap_epoch2': gap_epoch2
    }


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 4: Variable-depth MLP on MNIST at fixed MI'
    )
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of hidden layers (1-11)')
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
    parser.add_argument('--pixel_row', type=int, default=14,
                        help='Row for special pixel (0-27, default: 14)')
    parser.add_argument('--pixel_col', type=int, default=14,
                        help='Column for special pixel (0-27, default: 14)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--target_train_acc', type=float, default=99.0,
                        help='Target train accuracy to stop training')
    parser.add_argument('--initial_hidden_dim', type=int, default=1024,
                        help='Reference neurons for 1-layer network (determines param budget)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Calculate neurons per layer to match parameter budget
    # Formula accounts for bias terms:
    # 1 layer: (784+1)*N + (N+1)*10 = 795*N + 10
    # L layers: (784+1)*Nl + (L-1)*(Nl+1)*Nl + (Nl+1)*10
    #         = 785*Nl + (L-1)*Nl^2 + (L-1)*Nl + 10*Nl + 10
    #         = (L-1)*Nl^2 + (794+L)*Nl + 10
    # Setting equal: 795*N + 10 = (L-1)*Nl^2 + (794+L)*Nl + 10
    # Simplifying: (L-1)*Nl^2 + (794+L)*Nl - 795*N = 0
    # Nl = (-(794+L) + sqrt((794+L)^2 + 4*(L-1)*795*N)) / (2*(L-1))

    N_reference = args.initial_hidden_dim
    L = args.n_layers

    if L == 1:
        neurons_per_layer = N_reference
    else:
        a = L - 1
        b = 794 + L
        c = -795 * N_reference
        discriminant = b**2 - 4*a*c
        neurons_per_layer = round((-b + np.sqrt(discriminant)) / (2*a))

    # Build model (without BatchNorm for under-parameterization study)
    model = MLP_Variable(num_classes=10, n_layers=args.n_layers, initial_hidden_dim=neurons_per_layer,
                         with_bn=False)

    # Force special pixel to be always correct (no noise)
    noise_level = 0.0

    # Get hidden dims for display
    hidden_dims_str = f"{neurons_per_layer} × {args.n_layers}"
    reference_params = 795 * N_reference + 10  # Including bias terms

    print(f"\n{'='*80}")
    print(f"Experiment 4: MLP Variable Depth on MNIST")
    print(f"{'='*80}")
    print(f"Architecture:     MLP (Constant param budget)")
    print(f"Reference:        1 layer × {N_reference} neurons = {reference_params:,} params")
    print(f"Hidden Layers:    {args.n_layers}")
    print(f"Neurons/Layer:    {neurons_per_layer}")
    print(f"Hidden Dims:      {hidden_dims_str}")
    print(f"Total Params:     {model.count_parameters():,}")
    print(f"Target MI:        {args.mi_bits} bits")
    print(f"Noise Level:      {noise_level:.4f}")
    print(f"Seed:             {args.seed}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Max Epochs:       {args.max_epochs}")
    print(f"Target Train Acc: {args.target_train_acc}%")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Load data
    trainloader, testloader = get_mnist_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=noise_level,
        seed=args.seed,
        pixel_location=(args.pixel_row, args.pixel_col),
        augment=(not args.no_augment)
    )

    # Train model with occlusion sensitivity computation
    metrics = train_model(
        model, trainloader, testloader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, target_train_acc=args.target_train_acc,
        compute_occlusion=True
    )

    # Prepare results
    is_valid = metrics['final_train_acc'] >= args.target_train_acc

    result = {
        'model': 'MLP',
        'n_layers': args.n_layers,
        'hidden_dim': model.hidden_dim,
        'noise_level': noise_level,
        'mi_bits': args.mi_bits,
        'seed': args.seed,
        'valid': is_valid,
        'epochs_to_99pct': metrics['epochs_to_99pct'],
        'total_epochs': metrics['total_epochs'],
        'num_parameters': model.count_parameters(),
        'train_acc': metrics['final_train_acc'],
        'test_acc': metrics['final_test_acc'],
        'generalization_gap': metrics['generalization_gap'],
        'train_loss': metrics['final_train_loss'],
        'test_loss': metrics['final_test_loss'],
        'gap_epoch2': metrics['gap_epoch2'],
    }

    # Add occlusion data if available
    if metrics['occlusion_epoch2'] is not None:
        result.update({
            'occlusion_maps_epoch2': metrics['occlusion_epoch2']['occlusion_maps'],
            'sample_images_epoch2': metrics['occlusion_epoch2']['sample_images'],
            'sample_labels_epoch2': metrics['occlusion_epoch2']['sample_labels'],
        })
    if metrics['occlusion_final'] is not None:
        result.update({
            'occlusion_maps_final': metrics['occlusion_final']['occlusion_maps'],
            'sample_images_final': metrics['occlusion_final']['sample_images'],
            'sample_labels_final': metrics['occlusion_final']['sample_labels'],
        })

    if is_valid:
        print(f"\n{'='*80}")
        print(f"Training Complete (Valid - Reached {args.target_train_acc}%)")
        print(f"{'='*80}")
        print(f"Epochs to {args.target_train_acc}%: {metrics['epochs_to_99pct']}")
        print(f"Train Accuracy:   {metrics['final_train_acc']:.2f}%")
        print(f"Test Accuracy:    {metrics['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Training Complete (Invalid - Did Not Reach {args.target_train_acc}%)")
        print(f"{'='*80}")
        print(f"Total Epochs:     {metrics['total_epochs']}")
        print(f"Best Train Acc:   {metrics['final_train_acc']:.2f}%")
        print(f"Test Acc at Best: {metrics['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming convention: mlp_layers{n}_seed{seed}_results.npz
    save_path = f"{args.output_dir}/mlp_layers{args.n_layers}_seed{args.seed}_results.npz"

    np.savez(save_path, **result)
    print(f"Results saved to: {save_path}\n")

    # =========================================================================
    # EXPERIMENT 2: Train WITHOUT special pixel
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"Starting Experiment WITHOUT Special Pixel")
    print(f"{'='*80}\n")

    # Build new model (fresh initialization, without BatchNorm, same neurons_per_layer)
    model_nopixel = MLP_Variable(num_classes=10, n_layers=args.n_layers, initial_hidden_dim=neurons_per_layer,
                                 with_bn=False)

    # Load data WITHOUT special pixel
    trainloader_nopixel, testloader_nopixel = get_mnist_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        augment=(not args.no_augment)
    )

    # Train model without special pixel with occlusion sensitivity computation
    metrics_nopixel = train_model(
        model_nopixel, trainloader_nopixel, testloader_nopixel, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, target_train_acc=args.target_train_acc,
        compute_occlusion=True
    )

    # Prepare results for no-pixel version
    is_valid_nopixel = metrics_nopixel['final_train_acc'] >= args.target_train_acc

    result_nopixel = {
        'model': 'MLP',
        'n_layers': args.n_layers,
        'hidden_dim': model_nopixel.hidden_dim,
        'noise_level': None,  # No special pixel
        'mi_bits': None,  # No special pixel
        'seed': args.seed,
        'valid': is_valid_nopixel,
        'epochs_to_99pct': metrics_nopixel['epochs_to_99pct'],
        'total_epochs': metrics_nopixel['total_epochs'],
        'num_parameters': model_nopixel.count_parameters(),
        'train_acc': metrics_nopixel['final_train_acc'],
        'test_acc': metrics_nopixel['final_test_acc'],
        'generalization_gap': metrics_nopixel['generalization_gap'],
        'train_loss': metrics_nopixel['final_train_loss'],
        'test_loss': metrics_nopixel['final_test_loss'],
        'gap_epoch2': metrics_nopixel['gap_epoch2'],
    }

    # Add occlusion data if available
    if metrics_nopixel['occlusion_epoch2'] is not None:
        result_nopixel.update({
            'occlusion_maps_epoch2': metrics_nopixel['occlusion_epoch2']['occlusion_maps'],
            'sample_images_epoch2': metrics_nopixel['occlusion_epoch2']['sample_images'],
            'sample_labels_epoch2': metrics_nopixel['occlusion_epoch2']['sample_labels'],
        })
    if metrics_nopixel['occlusion_final'] is not None:
        result_nopixel.update({
            'occlusion_maps_final': metrics_nopixel['occlusion_final']['occlusion_maps'],
            'sample_images_final': metrics_nopixel['occlusion_final']['sample_images'],
            'sample_labels_final': metrics_nopixel['occlusion_final']['sample_labels'],
        })

    if is_valid_nopixel:
        print(f"\n{'='*80}")
        print(f"Training Complete (No Pixel - Valid - Reached {args.target_train_acc}%)")
        print(f"{'='*80}")
        print(f"Epochs to {args.target_train_acc}%: {metrics_nopixel['epochs_to_99pct']}")
        print(f"Train Accuracy:   {metrics_nopixel['final_train_acc']:.2f}%")
        print(f"Test Accuracy:    {metrics_nopixel['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics_nopixel['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Training Complete (No Pixel - Invalid - Did Not Reach {args.target_train_acc}%)")
        print(f"{'='*80}")
        print(f"Total Epochs:     {metrics_nopixel['total_epochs']}")
        print(f"Best Train Acc:   {metrics_nopixel['final_train_acc']:.2f}%")
        print(f"Test Acc at Best: {metrics_nopixel['final_test_acc']:.2f}%")
        print(f"Gen. Gap:         {metrics_nopixel['generalization_gap']:.2f}%")
        print(f"{'='*80}\n")

    # Save no-pixel results
    save_path_nopixel = f"{args.output_dir}/mlp_layers{args.n_layers}_seed{args.seed}_nopixel_results.npz"
    np.savez(save_path_nopixel, **result_nopixel)
    print(f"No-pixel results saved to: {save_path_nopixel}\n")


if __name__ == "__main__":
    main()
