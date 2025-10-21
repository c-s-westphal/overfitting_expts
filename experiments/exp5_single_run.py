"""
Experiment 5: Variable-depth MLP models on CIFAR-10 and MNIST with MI evaluation.

Studies how the number of hidden layers affects generalization in MLPs
by masking neurons in the first hidden layer and computing mutual information.

No special pixel - uses standard CIFAR-10 or MNIST classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from sklearn.metrics import mutual_info_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_unified import MLP_Unified
from data.data_loader import get_cifar10_dataloaders, get_mnist_dataloaders


def cutmix_data(x, y, alpha=0.5):
    """Apply CutMix augmentation.

    Args:
        x: Input tensor (batch_size, C, H, W)
        y: Target labels (batch_size,)
        alpha: CutMix parameter (default: 0.5)

    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    indices = torch.randperm(x.size(0))
    shuffled_y = y[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y, shuffled_y, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss


def generate_all_neuron_subsets(n_neurons):
    """Generate all possible non-empty subsets of neurons.

    For small n_neurons (e.g., 8), this generates all 2^n - 1 possible subsets.
    Each mask has True = keep neuron, False = zero out neuron.

    Args:
        n_neurons: Number of neurons in the layer

    Returns:
        List of boolean masks, one for each non-empty subset
    """
    masks = []
    # Generate all possible subsets (2^n combinations)
    # Skip 0 (empty set) by starting from 1
    for i in range(1, 2**n_neurons):
        mask = np.zeros(n_neurons, dtype=bool)
        # Convert integer to binary representation
        for bit_pos in range(n_neurons):
            if i & (1 << bit_pos):
                mask[bit_pos] = True
        masks.append(mask)

    return masks


def generate_random_neuron_masks(n_neurons, n_masks, seed=42):
    """Generate random masks for neuron selection in first hidden layer.

    Each mask randomly selects between 1 and (n_neurons - 2) neurons to keep.
    Returns masks where True = keep neuron, False = zero out neuron.
    Mask shape: (n_neurons,)
    """
    np.random.seed(seed)
    masks = []

    # Maximum subset size: all neurons except 2
    max_subset_size = max(1, n_neurons - 2)

    for _ in range(n_masks):
        # Random subset size between 1 and max_subset_size
        subset_size = np.random.randint(1, max_subset_size + 1)

        # Create flat mask and randomly select neurons to KEEP
        mask = np.zeros(n_neurons, dtype=bool)
        selected_indices = np.random.choice(n_neurons, subset_size, replace=False)
        mask[selected_indices] = True

        masks.append(mask)

    return masks


class NeuronMaskingHook:
    """Hook to mask individual neurons in first hidden layer output."""
    def __init__(self, mask):
        """
        Args:
            mask: Boolean array where True = keep neuron, False = zero out neuron
                  Shape: (n_neurons,)
        """
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """Apply neuron-wise masking during forward pass."""
        # output shape: (batch, n_neurons)
        # mask shape: (n_neurons,)
        masked_output = output * self.mask.to(output.device)
        return masked_output


def get_predictions_and_labels(model, data_loader, device, mask=None, max_batches=0):
    """Get model predictions and true labels.

    Args:
        model: The model to evaluate
        data_loader: Data loader
        device: Device to use
        mask: Optional mask to apply at first hidden layer
        max_batches: Maximum number of batches to process (0 = all)

    Returns:
        (predictions, labels) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_labels = []

    hook_handle = None
    if mask is not None:
        hook_layer = model.get_first_hidden_layer()
        hook = NeuronMaskingHook(mask)
        hook_handle = hook_layer.register_forward_hook(hook)

    try:
        batches_processed = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

                batches_processed += 1
                if max_batches and batches_processed >= max_batches:
                    break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    return predictions, labels


def calculate_mutual_information(predictions, labels):
    """Calculate mutual information between predictions and true labels.

    Uses discrete mutual information: I(Y; predictions)
    Maximum MI is log2(10) ≈ 3.32 bits for 10 classes (CIFAR-10 and MNIST).
    """
    return mutual_info_score(labels, predictions)


def evaluate_first_layer_mi(model, eval_loader, device, n_subsets, seed=42, max_batches=0, use_all_subsets=False):
    """Evaluate MI difference between full and masked first hidden layer.

    Args:
        model: The model to evaluate
        eval_loader: Data loader for evaluation
        device: Device to use
        n_subsets: Number of random subsets to use (if use_all_subsets=False)
        seed: Random seed for subset generation
        max_batches: Maximum batches to evaluate (0 = all)
        use_all_subsets: If True, use all possible subsets for small networks (<=16 neurons)

    Returns:
        (mi_full, mean_mi_masked, mi_difference)
    """
    model.eval()

    # Determine first layer dimensions
    n_neurons = model.hidden_dim

    # Generate neuron masks
    if use_all_subsets and n_neurons <= 16:
        masks = generate_all_neuron_subsets(n_neurons)
        print(f"  Using all {len(masks)} possible subsets for {n_neurons} neurons", flush=True)
    else:
        masks = generate_random_neuron_masks(n_neurons, n_subsets, seed)
        print(f"  Using {len(masks)} random subsets for {n_neurons} neurons", flush=True)

    # Get predictions for full model
    full_predictions, labels = get_predictions_and_labels(
        model, eval_loader, device, mask=None, max_batches=max_batches
    )

    # Calculate MI for full model
    mi_full = calculate_mutual_information(full_predictions, labels)

    # Calculate MI for each masked version
    masked_mis = []
    for mask in masks:
        masked_predictions, _ = get_predictions_and_labels(
            model, eval_loader, device, mask=mask, max_batches=max_batches
        )
        mi_masked = calculate_mutual_information(masked_predictions, labels)
        masked_mis.append(mi_masked)

    # Calculate mean MI across all masks
    mean_mi_masked = np.mean(masked_mis)
    mi_difference = mi_full - mean_mi_masked

    return mi_full, mean_mi_masked, mi_difference


def evaluate_accuracy(model, data_loader, device, max_batches=0):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    batches_processed = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batches_processed += 1
            if max_batches and batches_processed >= max_batches:
                break

    return 100. * correct / total


def separate_parameters_for_weight_decay(model):
    """Separate parameters into groups for selective weight decay.

    Apply weight decay only to Linear weights, not to LayerNorm params or any biases.

    Returns:
        List of parameter dicts for optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay on biases or LayerNorm parameters
        if 'bias' in name or 'norm' in name or isinstance(param, nn.LayerNorm):
            no_decay_params.append(param)
        else:
            # Apply weight decay to Linear weights
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': None},  # Will be set by optimizer
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_cutmix=False, cutmix_alpha=0.5, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix if enabled
        if use_cutmix and np.random.rand() < 0.5:  # Apply CutMix with 50% probability
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Mixed loss for CutMix
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    print("Starting exp5_single_run.py main()...", flush=True)
    parser = argparse.ArgumentParser(
        description='Experiment 5: Variable-depth MLP on CIFAR-10/MNIST with MI evaluation'
    )
    print("Parser created", flush=True)
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of hidden layers (1-10)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                        help='Dataset to use (cifar10 or mnist)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability (default: 0.3)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for batch 512 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # MI evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate MI every N epochs (default: 10)')
    parser.add_argument('--n_masks_train', type=int, default=20,
                        help='Number of masks for training MI evaluation (default: 20)')
    parser.add_argument('--n_masks_final', type=int, default=40,
                        help='Number of masks for final MI evaluation (default: 40)')
    parser.add_argument('--max_eval_batches_train', type=int, default=20,
                        help='Max batches for training MI evaluation (default: 20)')
    parser.add_argument('--max_eval_batches_final', type=int, default=40,
                        help='Max batches for final MI evaluation (default: 40)')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    print("Parsing arguments...", flush=True)
    args = parser.parse_args()
    print(f"Arguments parsed: dataset={args.dataset}, n_layers={args.n_layers}, seed={args.seed}", flush=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    print("Setting device...", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Scale learning rate based on batch size (linear scaling)
    scaled_lr = args.lr * (args.batch_size / 512)

    # Determine dataset-specific parameters
    if args.dataset == 'cifar10':
        input_dim = 3072  # 3x32x32
        num_classes = 10
    elif args.dataset == 'mnist':
        input_dim = 784  # 1x28x28
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create model
    model = MLP_Unified(
        num_classes=num_classes,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        input_dim=input_dim
    )
    model = model.to(device)

    print(f"\n{'='*80}")
    print(f"Experiment 5: MLP Variable Depth on {args.dataset.upper()}")
    print(f"{'='*80}")
    print(f"Dataset:          {args.dataset.upper()}")
    print(f"Architecture:     MLP with LayerNorm and residual connections")
    print(f"Input Dim:        {input_dim}")
    print(f"Hidden Layers:    {args.n_layers}")
    print(f"Hidden Dim:       {args.hidden_dim}")
    print(f"Dropout:          {args.dropout}")
    print(f"Total Params:     {model.count_parameters():,}")
    print(f"Seed:             {args.seed}")
    print(f"Train Subset:     First 10,000 training images (no augmentation)")
    print(f"Target Train Acc: 99.99%")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {scaled_lr:.6f} (scaled from {args.lr} for batch {args.batch_size})")
    print(f"Weight Decay:     {args.weight_decay}")
    print(f"Grad Clip:        {args.grad_clip}")
    print(f"Warmup Epochs:    {args.warmup_epochs}")
    print(f"Label Smoothing:  {args.label_smoothing}")
    print(f"Max Epochs:       {args.epochs}")
    print(f"Device:           {args.device}")
    print(f"{'='*80}\n")

    # Create data loaders with dataset-specific augmentation
    print("\nCreating data loaders...", flush=True)
    from torchvision import datasets, transforms

    # Training subset size: first 10,000 images
    train_subset_size = 10000
    train_indices = list(range(train_subset_size))

    if args.dataset == 'cifar10':
        # CIFAR-10: No augmentation, clean images only
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load datasets
        train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    elif args.dataset == 'mnist':
        # MNIST: With augmentation for training, clean for evaluation
        print("Loading MNIST data...", flush=True)
        train_transform_aug = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
        ])
        train_transform_clean = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # Load datasets - two versions of training set
        train_dataset_aug = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform_aug)
        train_dataset_clean = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform_clean)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        print("MNIST data loaded", flush=True)

    print(f"Creating training subset with first {train_subset_size} images...", flush=True)
    # Create subset of first 10,000 training images
    if args.dataset == 'mnist':
        train_subset_aug = torch.utils.data.Subset(train_dataset_aug, train_indices)
        train_subset_clean = torch.utils.data.Subset(train_dataset_clean, train_indices)
    else:
        train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)

    # Create data loaders
    if args.dataset == 'mnist':
        # Train on augmented data
        train_loader = torch.utils.data.DataLoader(
            train_subset_aug, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        # Evaluate on clean data (no augmentation)
        eval_loader = torch.utils.data.DataLoader(
            train_subset_clean, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        eval_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"Data loaders created: train={train_subset_size} images, test={len(test_dataset)} images", flush=True)

    # Setup optimizer (AdamW with selective weight decay)
    print("Setting up optimizer...", flush=True)
    param_groups = separate_parameters_for_weight_decay(model)
    param_groups[0]['weight_decay'] = args.weight_decay  # Apply WD to Linear weights

    optimizer = optim.AdamW(
        param_groups,
        lr=scaled_lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Setup learning rate scheduler: warmup + cosine annealing to 1e-5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of base LR
        end_factor=1.0,     # Reach full LR
        total_iters=args.warmup_epochs
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-5
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # Tracking
    mi_history = []
    train_acc_history = []
    test_acc_history = []
    epochs_evaluated = []
    epoch_reached_target = None

    # Training loop
    target_train_acc = 99.0
    print(f"\nStarting training (target: {target_train_acc}% train acc, max: {args.epochs} epochs)...")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        # Train (no augmentation)
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_cutmix=False, cutmix_alpha=0.5, grad_clip=args.grad_clip
        )

        # Step scheduler
        scheduler.step()

        # Evaluate train accuracy
        train_acc = evaluate_accuracy(model, eval_loader, device)

        # Check if target reached
        if train_acc >= target_train_acc and epoch_reached_target is None:
            epoch_reached_target = epoch
            print(f"\n{'='*70}", flush=True)
            print(f"Target train accuracy {target_train_acc}% reached at epoch {epoch}!", flush=True)
            print(f"{'='*70}\n", flush=True)

        # Evaluate MI and print progress at specified intervals
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            # Evaluate test accuracy
            test_acc = evaluate_accuracy(model, test_loader, device)

            print(f"Epoch {epoch}/{args.epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%", flush=True)

            # Evaluate MI (skip at final epoch since we do comprehensive MI eval after loop)
            if epoch != args.epochs:
                print(f"  Evaluating MI on train subset (n_masks={args.n_masks_train}, max_batches={args.max_eval_batches_train})...", flush=True)
                mi_full, mean_mi_masked, mi_diff = evaluate_first_layer_mi(
                    model, eval_loader, device,
                    n_subsets=args.n_masks_train,
                    seed=args.seed + epoch,  # Different seed each time
                    max_batches=args.max_eval_batches_train,
                    use_all_subsets=False  # Use random subsets during training for speed
                )
                print(f"  MI: {mi_full:.6f}, MI_masked: {mean_mi_masked:.6f}, MI_diff: {mi_diff:.6f}", flush=True)

                mi_history.append(mi_diff)
                train_acc_history.append(train_acc)
                test_acc_history.append(test_acc)
                epochs_evaluated.append(epoch)

        # Early stopping if target accuracy reached
        if train_acc >= target_train_acc:
            print(f"\n{'='*70}", flush=True)
            print(f"Stopping early: Target train accuracy {target_train_acc}% reached at epoch {epoch}", flush=True)
            print(f"{'='*70}\n", flush=True)
            break

    print("\n" + "="*70)
    if epoch_reached_target is not None:
        print(f"Training completed! Target accuracy reached at epoch {epoch_reached_target}")
    else:
        print(f"Training completed! Maximum epochs ({args.epochs}) reached")
        print(f"Final train accuracy: {train_acc:.2f}%")
    print("="*70)

    # Final MI evaluation with all possible subsets and all batches (on train subset)
    print(f"\nFinal MI evaluation on train subset (all subsets, all batches)...", flush=True)
    final_mi_full, final_mean_mi_masked, final_mi_diff = evaluate_first_layer_mi(
        model, eval_loader, device,
        n_subsets=args.n_masks_final,  # Ignored when use_all_subsets=True
        seed=args.seed,
        max_batches=0,  # Use all batches for final evaluation
        use_all_subsets=True  # Use all possible subsets for final evaluation
    )

    print(f"Final MI: {final_mi_full:.6f}", flush=True)
    print(f"Final MI_masked: {final_mean_mi_masked:.6f}", flush=True)
    print(f"Final MI_diff: {final_mi_diff:.6f}", flush=True)

    # Final accuracy evaluation
    final_train_acc = evaluate_accuracy(model, eval_loader, device)
    final_test_acc = evaluate_accuracy(model, test_loader, device)
    final_gen_gap = final_train_acc - final_test_acc

    print(f"\nFinal Results:", flush=True)
    print(f"  Train Acc: {final_train_acc:.2f}%", flush=True)
    print(f"  Test Acc:  {final_test_acc:.2f}%", flush=True)
    print(f"  Gen Gap:   {final_gen_gap:.2f}%", flush=True)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    save_path = f"{args.output_dir}/mlp_{args.dataset}_layers{args.n_layers}_seed{args.seed}_results.npz"

    result = {
        'model': f'MLP_Unified',
        'dataset': args.dataset,
        'n_layers': args.n_layers,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'seed': args.seed,
        'num_parameters': model.count_parameters(),
        'max_epochs': args.epochs,
        'target_train_acc': target_train_acc,
        'epochs_trained': epoch,
        'epoch_reached_target': epoch_reached_target if epoch_reached_target is not None else -1,
        'batch_size': args.batch_size,
        'lr': scaled_lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'warmup_epochs': args.warmup_epochs,
        'label_smoothing': args.label_smoothing,
        'train_subset_size': train_subset_size,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'final_gen_gap': final_gen_gap,
        'final_mi_full': final_mi_full,
        'final_mean_mi_masked': final_mean_mi_masked,
        'final_mi_diff': final_mi_diff,
        'epochs_evaluated': np.array(epochs_evaluated),
        'mi_history': np.array(mi_history),
        'train_acc_history': np.array(train_acc_history),
        'test_acc_history': np.array(test_acc_history),
    }

    np.savez(save_path, **result)
    print(f"\nResults saved to: {save_path}\n", flush=True)


if __name__ == "__main__":
    main()
