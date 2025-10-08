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


def train_epoch(model, dataloader, criterion, optimizer, device,
                log_gradients=False, grad_monitor_epochs=10, epoch_index=1,
                monitor_every_n_batches=50):
    """Train for one epoch with optional gradient monitoring."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Optional gradient monitoring for early epochs
        if log_gradients and epoch_index <= grad_monitor_epochs and (batch_index % monitor_every_n_batches == 0 or batch_index == 1):
            with torch.no_grad():
                total_grad_norm = 0.0
                first_last = {}
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    # Track a few key layers
                    if any(key in name for key in ('.0.weight', '.2.weight', 'classifier.0.weight', 'classifier.1.weight', 'classifier.2.weight')):
                        first_last[name] = param_norm
                total_grad_norm = total_grad_norm ** 0.5
                print(f"[GradMon] epoch={epoch_index} batch={batch_index} total_grad_L2={total_grad_norm:.4e}")
                for k, v in first_last.items():
                    print(f"  - {k}: {v:.4e}")

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
    """Get learning rate for given epoch (1-indexed).

    LR schedule: drops at epoch 100, 150, then every 50 epochs thereafter.
    - Epochs 1-100: 0.1
    - Epochs 101-150: 0.01
    - Epochs 151-200: 0.001
    - Epochs 201-250: 0.0001
    - Epochs 251-300: 0.00001
    - And continues dropping by 10x every 50 epochs
    """
    if epoch <= 100:
        return 0.1
    elif epoch <= 150:
        return 0.01
    else:
        # After epoch 150, drop by 10x every 50 epochs
        periods_after_150 = (epoch - 151) // 50 + 1
        return 0.01 * (0.1 ** periods_after_150)


def train_model_adaptive(model, trainloader, testloader, device='cuda',
                         base_lr=0.1, weight_decay=5e-4, optimizer_type='sgd',
                         warmup_epochs=0, log_gradients=False, grad_monitor_epochs=10,
                         log_activations=False, act_monitor_epochs=10, monitor_every_n_batches=50,
                         initial_epochs=200, max_epochs=500):
    """
    Train model with adaptive duration (200-500 epochs).

    Trains for 200 epochs initially. If train_acc < 99%, extends to 500 epochs.
    Learning rate schedule based on current epoch: 0.1 (1-100), 0.01 (101-150), 0.001 (151+).

    Returns:
        dict: Training metrics including final accuracies and generalization gap
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # Initial training
    print(f"\nTraining for {initial_epochs} epochs...")
    pbar = tqdm(range(1, initial_epochs + 1), desc="Training")

    # Optional activation monitoring via forward hooks
    hooks = []
    activation_stats = {}
    if log_activations:
        def make_hook(name):
            def hook_fn(module, inp, out):
                with torch.no_grad():
                    if isinstance(out, tuple):
                        tensor = out[0]
                    else:
                        tensor = out
                    tensor = tensor.detach()
                    zero_frac = (tensor <= 0).float().mean().item()
                    mean_val = tensor.mean().item()
                    std_val = tensor.std(unbiased=False).item()
                    activation_stats.setdefault(name, []).append((zero_frac, mean_val, std_val))
            return hook_fn
        # Attach to ReLU activations to detect dying ReLUs
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(make_hook(name)))

    for epoch in pbar:
        # Update learning rate based on current epoch with optional warmup
        scheduled_lr = get_lr_for_epoch(epoch)
        warmup_factor = 1.0
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, warmup_epochs))
        current_lr = scheduled_lr * warmup_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Clear activation stats at start of epoch for logging window
        if log_activations and epoch <= act_monitor_epochs:
            activation_stats.clear()

        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device,
            log_gradients=log_gradients, grad_monitor_epochs=grad_monitor_epochs,
            epoch_index=epoch, monitor_every_n_batches=monitor_every_n_batches
        )
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

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{initial_epochs} | LR: {current_lr} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        # Print activation stats for early epochs
        if log_activations and epoch <= act_monitor_epochs:
            # Aggregate per-layer stats across batches in this epoch
            print(f"[ActMon] epoch={epoch} stats for ReLU activations (zero_frac, mean, std)")
            for name, records in activation_stats.items():
                if not records:
                    continue
                zf = np.mean([r[0] for r in records])
                mv = np.mean([r[1] for r in records])
                sv = np.mean([r[2] for r in records])
                print(f"  - {name}: zero_frac={zf:.3f}, mean={mv:.4f}, std={sv:.4f}")

    # Check if we need to extend training
    final_train_acc = train_accs[-1]
    if final_train_acc < 99.0:
        print(f"\nTrain accuracy {final_train_acc:.2f}% < 99%. Extending to {max_epochs} epochs...")
        pbar = tqdm(range(initial_epochs + 1, max_epochs + 1), desc="Extended Training")

        for epoch in pbar:
            # Update learning rate based on current epoch with optional warmup
            scheduled_lr = get_lr_for_epoch(epoch)
            warmup_factor = 1.0
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_factor = float(epoch) / float(max(1, warmup_epochs))
            current_lr = scheduled_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            if log_activations and epoch <= act_monitor_epochs:
                activation_stats.clear()

            train_loss, train_acc = train_epoch(
                model, trainloader, criterion, optimizer, device,
                log_gradients=log_gradients, grad_monitor_epochs=grad_monitor_epochs,
                epoch_index=epoch, monitor_every_n_batches=monitor_every_n_batches
            )
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

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}/{max_epochs} | LR: {current_lr} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

            if log_activations and epoch <= act_monitor_epochs:
                print(f"[ActMon] epoch={epoch} stats for ReLU activations (zero_frac, mean, std)")
                for name, records in activation_stats.items():
                    if not records:
                        continue
                    zf = np.mean([r[0] for r in records])
                    mv = np.mean([r[1] for r in records])
                    sv = np.mean([r[2] for r in records])
                    print(f"  - {name}: zero_frac={zf:.3f}, mean={mv:.4f}, std={sv:.4f}")

            # Early stop if we reach 99% during extended training
            if train_acc >= 99.0:
                print(f"\nReached 99% train accuracy at epoch {epoch}. Stopping training.")
                break

    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]

    # Remove hooks
    for h in hooks:
        h.remove()

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
    parser.add_argument('--pixel_row', type=int, default=0,
                        help='Row for special pixel (0-31)')
    parser.add_argument('--pixel_col', type=int, default=0,
                        help='Column for special pixel (0-31)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable train-time augmentation')
    parser.add_argument('--with_bn', action='store_true',
                        help='Enable BatchNorm in variable VGG')
    parser.add_argument('--dropout_p', type=float, default=0.5,
                        help='Dropout probability in classifier')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--base_lr', type=float, default=0.1,
                        help='Base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of LR warmup epochs')
    parser.add_argument('--log_gradients', action='store_true',
                        help='Log gradient norms in early epochs')
    parser.add_argument('--grad_monitor_epochs', type=int, default=10,
                        help='Number of epochs to log gradient norms')
    parser.add_argument('--log_activations', action='store_true',
                        help='Log activation zero fraction/mean/std in early epochs')
    parser.add_argument('--act_monitor_epochs', type=int, default=10,
                        help='Number of epochs to log activation stats')
    parser.add_argument('--monitor_every_n_batches', type=int, default=50,
                        help='How often to log per-epoch monitors (in batches)')
    parser.add_argument('--initial_epochs', type=int, default=200,
                        help='Initial training epochs before optional extension')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum epochs if train acc < 99% after initial epochs')

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
    model = model_fn(num_classes=10, n_layers=args.n_layers, with_bn=args.with_bn, dropout_p=args.dropout_p)

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
        seed=args.seed,
        pixel_location=(args.pixel_row, args.pixel_col),
        augment=(not args.no_augment)
    )

    # Train model with adaptive duration
    metrics = train_model_adaptive(
        model, trainloader, testloader, device=args.device,
        base_lr=args.base_lr, weight_decay=args.weight_decay, optimizer_type=args.optimizer,
        warmup_epochs=args.warmup_epochs, log_gradients=args.log_gradients,
        grad_monitor_epochs=args.grad_monitor_epochs, log_activations=args.log_activations,
        act_monitor_epochs=args.act_monitor_epochs, monitor_every_n_batches=args.monitor_every_n_batches,
        initial_epochs=args.initial_epochs, max_epochs=args.max_epochs
    )

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
