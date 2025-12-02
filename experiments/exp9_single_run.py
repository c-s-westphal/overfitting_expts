"""
Experiment 9: Pruning ratio and MI analysis for binary MNIST classification.

This experiment:
1. Trains a small MLP (5 hidden layers, 5 neurons each) on MNIST binary (0 vs 1)
2. For each layer, computes:
   - Allowed pruning ratio: min neurons needed to recover train accuracy
     (freeze layer & earlier layers, retrain later layers)
   - MI between all neuron subsets (sizes 1-4) and output probability using KDE
   - Entropy of output probability
3. Stores per-layer: entropy, average MI across subsets, allowed pruning ratio
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import copy
from itertools import combinations
from tqdm import tqdm
from scipy.stats import gaussian_kde

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_mnist_binary_dataloaders


class MLP_Binary(nn.Module):
    """
    Small MLP for binary MNIST classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - 5 hidden layers with 5 neurons each
    - Output: 2 classes (digits 0 and 1)
    """
    def __init__(self, n_layers=5, neurons_per_layer=5, num_classes=2):
        super(MLP_Binary, self).__init__()

        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_dim = 784

        # Build layers as a ModuleList for easy access
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # First hidden layer: 784 -> neurons_per_layer
        self.hidden_layers.append(nn.Linear(self.input_dim, neurons_per_layer))
        self.activations.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.activations.append(nn.ReLU())

        # Output layer
        self.output_layer = nn.Linear(neurons_per_layer, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
        return self.output_layer(x)

    def forward_with_activations(self, x):
        """Forward pass returning activations at each hidden layer."""
        x = x.view(x.size(0), -1)
        activations = []
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
            activations.append(x)
        output = self.output_layer(x)
        return output, activations

    def get_output_probability(self, x):
        """Get probability of class 1 (digit 1)."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # p(class=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
                lr=0.001, weight_decay=1e-4, max_epochs=200, target_train_acc=100.0):
    """
    Train model until target train accuracy is reached.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nTraining until {target_train_acc}% train accuracy...")
    pbar = tqdm(range(1, max_epochs + 1), desc="Training")

    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        pbar.set_postfix({
            'Train': f'{train_acc:.2f}%',
            'Test': f'{test_acc:.2f}%'
        })

        if train_acc >= target_train_acc:
            print(f"\nReached {target_train_acc}% train accuracy at epoch {epoch}")
            return {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epochs': epoch
            }

    print(f"\nDid not reach {target_train_acc}% after {max_epochs} epochs. Final: {train_acc:.2f}%")
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'epochs': max_epochs
    }


def get_layer_neuron_weights(model, layer_idx):
    """
    Get weight magnitudes for neurons in a specific layer.

    Returns importance scores based on L2 norm of outgoing weights.
    """
    layer = model.hidden_layers[layer_idx]
    # Use L2 norm of weight rows (each row corresponds to one output neuron)
    weight_norms = torch.norm(layer.weight, p=2, dim=1).detach().cpu().numpy()
    return weight_norms


def create_pruned_model(original_model, layer_idx, keep_neurons, device):
    """
    Create a new model with a pruned layer.

    The pruned layer keeps only the specified neurons.
    Layers before and including the pruned layer are frozen.
    Layers after are reinitialized for retraining.

    Args:
        original_model: The trained model
        layer_idx: Which layer to prune (0-indexed)
        keep_neurons: List of neuron indices to keep
        device: Device to use

    Returns:
        New model with modified architecture
    """
    n_keep = len(keep_neurons)
    n_layers = original_model.n_layers
    neurons_per_layer = original_model.neurons_per_layer

    # Create new model structure
    class PrunedMLP(nn.Module):
        def __init__(self):
            super(PrunedMLP, self).__init__()
            self.hidden_layers = nn.ModuleList()
            self.activations = nn.ModuleList()

            # Layers before pruned layer: same structure, frozen
            for i in range(layer_idx):
                self.hidden_layers.append(nn.Linear(
                    784 if i == 0 else neurons_per_layer,
                    neurons_per_layer
                ))
                self.activations.append(nn.ReLU())

            # Pruned layer: reduced output neurons
            self.hidden_layers.append(nn.Linear(
                784 if layer_idx == 0 else neurons_per_layer,
                n_keep
            ))
            self.activations.append(nn.ReLU())

            # Layers after pruned layer: start with n_keep input, then back to normal
            for i in range(layer_idx + 1, n_layers):
                in_dim = n_keep if i == layer_idx + 1 else neurons_per_layer
                self.hidden_layers.append(nn.Linear(in_dim, neurons_per_layer))
                self.activations.append(nn.ReLU())

            # Output layer
            out_in_dim = n_keep if layer_idx == n_layers - 1 else neurons_per_layer
            self.output_layer = nn.Linear(out_in_dim, 2)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            for layer, act in zip(self.hidden_layers, self.activations):
                x = act(layer(x))
            return self.output_layer(x)

    pruned_model = PrunedMLP().to(device)

    # Copy weights for frozen layers (before pruned layer)
    with torch.no_grad():
        for i in range(layer_idx):
            pruned_model.hidden_layers[i].weight.copy_(original_model.hidden_layers[i].weight)
            pruned_model.hidden_layers[i].bias.copy_(original_model.hidden_layers[i].bias)
            # Freeze
            pruned_model.hidden_layers[i].weight.requires_grad = False
            pruned_model.hidden_layers[i].bias.requires_grad = False

        # Copy weights for pruned layer (only keep_neurons)
        orig_layer = original_model.hidden_layers[layer_idx]
        pruned_model.hidden_layers[layer_idx].weight.copy_(orig_layer.weight[keep_neurons, :])
        pruned_model.hidden_layers[layer_idx].bias.copy_(orig_layer.bias[keep_neurons])
        # Freeze the pruned layer
        pruned_model.hidden_layers[layer_idx].weight.requires_grad = False
        pruned_model.hidden_layers[layer_idx].bias.requires_grad = False

    # Layers after pruned layer are reinitialized (already done by default)
    # They remain trainable

    return pruned_model


def compute_pruning_ratio(model, layer_idx, trainloader, device,
                          target_train_acc, retrain_epochs=10, lr=0.001):
    """
    Compute the minimum number of neurons needed to recover train accuracy.

    Process:
    1. Get neuron importance by weight magnitude
    2. Start with 1 neuron, try to recover accuracy by retraining later layers
    3. If fails, try 2, 3, ... until success

    Returns:
        min_neurons: Minimum neurons needed
        pruning_ratio: (5 - min_neurons) / 5 = fraction that can be pruned
    """
    neurons_per_layer = model.neurons_per_layer

    # Get neuron importance
    importance = get_layer_neuron_weights(model, layer_idx)
    sorted_indices = np.argsort(importance)[::-1]  # Descending order

    print(f"\n  Testing layer {layer_idx + 1} pruning...")

    for n_keep in range(1, neurons_per_layer + 1):
        keep_neurons = sorted_indices[:n_keep].tolist()

        # Create pruned model
        pruned_model = create_pruned_model(model, layer_idx, keep_neurons, device)

        # Retrain layers after the pruned layer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, pruned_model.parameters()),
            lr=lr
        )

        best_train_acc = 0.0
        for epoch in range(retrain_epochs):
            train_loss, train_acc = train_epoch(
                pruned_model, trainloader, criterion, optimizer, device
            )
            best_train_acc = max(best_train_acc, train_acc)

            if train_acc >= target_train_acc:
                break

        print(f"    {n_keep} neurons: best train acc = {best_train_acc:.2f}%")

        if best_train_acc >= target_train_acc:
            pruning_ratio = (neurons_per_layer - n_keep) / neurons_per_layer
            print(f"    -> Success! Min neurons = {n_keep}, pruning ratio = {pruning_ratio:.2f}")
            return n_keep, pruning_ratio

    # Should not reach here if original model achieves target
    print(f"    -> Could not recover accuracy even with all neurons!")
    return neurons_per_layer, 0.0


def collect_activations_and_outputs(model, dataloader, device):
    """
    Collect all hidden layer activations and output probabilities on training set.

    Returns:
        activations: List of arrays, one per layer, shape (N, neurons_per_layer)
        output_probs: Array of p(class=1), shape (N,)
    """
    model.eval()
    all_activations = [[] for _ in range(model.n_layers)]
    all_output_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            output, acts = model.forward_with_activations(inputs)
            probs = torch.softmax(output, dim=1)[:, 1]  # p(class=1)

            for i, act in enumerate(acts):
                all_activations[i].append(act.cpu().numpy())
            all_output_probs.append(probs.cpu().numpy())

    # Concatenate
    activations = [np.concatenate(a, axis=0) for a in all_activations]
    output_probs = np.concatenate(all_output_probs, axis=0)

    return activations, output_probs


def add_jitter(data, jitter_scale=1e-6):
    """
    Add small jitter to data to prevent singular covariance matrices.

    For dimensions with zero/near-zero variance (dead neurons),
    adds small Gaussian noise proportional to the data range.
    """
    data = data.copy()
    for i in range(data.shape[1]):
        col_std = np.std(data[:, i])
        if col_std < 1e-10:
            # Dead neuron: add small noise based on data scale
            data_range = np.max(np.abs(data[:, i])) + 1e-10
            data[:, i] += np.random.normal(0, data_range * jitter_scale, size=data.shape[0])
        else:
            # Add tiny jitter proportional to std
            data[:, i] += np.random.normal(0, col_std * jitter_scale, size=data.shape[0])
    return data


def check_dead_neurons(X):
    """Check which neurons have zero variance (dead neurons)."""
    if X.ndim == 1:
        return np.std(X) < 1e-10
    return np.std(X, axis=0) < 1e-10


def compute_kde_entropy(data, n_samples=10000):
    """
    Compute entropy using KDE.

    H(X) = -E[log p(X)]

    Uses Monte Carlo estimation by sampling from the KDE.
    """
    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Add jitter to handle singular covariance
    data_jittered = add_jitter(data)

    # KDE expects (n_features, n_samples)
    data_T = data_jittered.T

    try:
        kde = gaussian_kde(data_T)

        # Sample from KDE and estimate entropy
        samples = kde.resample(n_samples)
        log_probs = kde.logpdf(samples)

        # Entropy = -E[log p(x)]
        entropy = -np.mean(log_probs)
        return entropy
    except Exception as e:
        print(f"    KDE entropy failed: {e}")
        return np.nan


def compute_kde_mi(X, Y, n_samples=10000):
    """
    Compute MI between X (neuron activations) and Y (output probability) using KDE.

    MI(X; Y) = H(X) + H(Y) - H(X, Y)

    Args:
        X: Array of shape (N, d) - neuron activations for subset
        Y: Array of shape (N,) - output probabilities
        n_samples: Number of samples for Monte Carlo estimation

    Returns:
        Mutual information estimate
    """
    # Ensure proper shapes
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    # Check for dead neurons and report
    dead_mask = check_dead_neurons(X)
    if np.any(dead_mask):
        n_dead = np.sum(dead_mask)
        # If ALL neurons in subset are dead, MI is 0 (no information)
        if n_dead == X.shape[1]:
            return 0.0

    # Add jitter to handle singular covariance
    X_jittered = add_jitter(X)
    Y_jittered = add_jitter(Y)

    # Joint data
    XY = np.hstack([X_jittered, Y_jittered])

    try:
        # Fit KDEs
        kde_x = gaussian_kde(X_jittered.T)
        kde_y = gaussian_kde(Y_jittered.T)
        kde_xy = gaussian_kde(XY.T)

        # Sample from joint distribution
        samples_xy = kde_xy.resample(n_samples)

        # Compute log probabilities
        log_p_xy = kde_xy.logpdf(samples_xy)
        log_p_x = kde_x.logpdf(samples_xy[:X.shape[1], :])
        log_p_y = kde_y.logpdf(samples_xy[X.shape[1]:, :])

        # MI = E[log p(x,y) - log p(x) - log p(y)]
        mi = np.mean(log_p_xy - log_p_x - log_p_y)

        return max(0, mi)  # MI should be non-negative
    except Exception as e:
        print(f"    KDE MI failed: {e}")
        return np.nan


def compute_layer_mi_stats(activations, output_probs, neurons_per_layer=5):
    """
    Compute MI statistics for all subsets of neurons in a layer.

    Computes MI for all subsets of size 1 to neurons_per_layer-1 (30 subsets for 5 neurons).

    Returns:
        avg_mi: Average MI across all subsets
        all_mi: Dict mapping subset tuple to MI value
        output_entropy: Entropy of output probability
    """
    # Compute output entropy once
    output_entropy = compute_kde_entropy(output_probs)
    print(f"    Output entropy: {output_entropy:.4f}")

    all_mi = {}
    mi_values = []

    # Generate all subsets of size 1 to neurons_per_layer-1
    neuron_indices = list(range(neurons_per_layer))

    for subset_size in range(1, neurons_per_layer):  # 1, 2, 3, 4
        for subset in combinations(neuron_indices, subset_size):
            # Extract activations for this subset
            subset_activations = activations[:, list(subset)]

            # Compute MI
            mi = compute_kde_mi(subset_activations, output_probs)
            all_mi[subset] = mi
            mi_values.append(mi)

            print(f"    Subset {subset}: MI = {mi:.4f}")

    avg_mi = np.nanmean(mi_values)
    print(f"    Average MI across all subsets: {avg_mi:.4f}")

    return avg_mi, all_mi, output_entropy


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 9: Pruning ratio and MI analysis for binary MNIST'
    )
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of hidden layers (default: 5)')
    parser.add_argument('--neurons_per_layer', type=int, default=5,
                        help='Neurons per hidden layer (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum epochs for initial training')
    parser.add_argument('--retrain_epochs', type=int, default=10,
                        help='Epochs for retraining after pruning')
    parser.add_argument('--target_train_acc', type=float, default=100.0,
                        help='Target train accuracy')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"\n{'='*80}")
    print(f"Experiment 9: Pruning Ratio and MI Analysis")
    print(f"{'='*80}")
    print(f"Architecture:      MLP ({args.n_layers} layers × {args.neurons_per_layer} neurons)")
    print(f"Task:              MNIST Binary (0 vs 1)")
    print(f"Seed:              {args.seed}")
    print(f"Device:            {args.device}")
    print(f"Target Train Acc:  {args.target_train_acc}%")
    print(f"{'='*80}\n")

    # Create model
    model = MLP_Binary(
        n_layers=args.n_layers,
        neurons_per_layer=args.neurons_per_layer,
        num_classes=2
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # Load data
    trainloader, testloader = get_mnist_binary_dataloaders(
        batch_size=args.batch_size,
        num_workers=4
    )

    # Count samples
    n_train = sum(len(batch[0]) for batch in trainloader)
    n_test = sum(len(batch[0]) for batch in testloader)
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")

    # Train model
    train_metrics = train_model(
        model, trainloader, testloader, device=args.device,
        lr=args.lr, max_epochs=args.max_epochs,
        target_train_acc=args.target_train_acc
    )

    original_train_acc = train_metrics['train_acc']
    print(f"\nTraining complete. Train acc: {original_train_acc:.2f}%, Test acc: {train_metrics['test_acc']:.2f}%")

    # Collect activations and output probabilities
    print(f"\n{'='*80}")
    print("Collecting activations and output probabilities...")
    print(f"{'='*80}")

    activations, output_probs = collect_activations_and_outputs(model, trainloader, args.device)
    print(f"Collected {len(output_probs)} samples")
    for i, act in enumerate(activations):
        print(f"  Layer {i+1}: shape {act.shape}")

    # Results storage
    results = {
        'seed': args.seed,
        'n_layers': args.n_layers,
        'neurons_per_layer': args.neurons_per_layer,
        'train_acc': original_train_acc,
        'test_acc': train_metrics['test_acc'],
        'epochs': train_metrics['epochs'],
        'layer_results': {}
    }

    # Per-layer analysis
    print(f"\n{'='*80}")
    print("Per-layer Analysis: Pruning Ratio and MI")
    print(f"{'='*80}")

    layer_pruning_ratios = []
    layer_avg_mis = []
    layer_entropies = []

    for layer_idx in range(args.n_layers):
        print(f"\n--- Layer {layer_idx + 1} ---")

        # Compute pruning ratio
        min_neurons, pruning_ratio = compute_pruning_ratio(
            model, layer_idx, trainloader, args.device,
            target_train_acc=original_train_acc,
            retrain_epochs=args.retrain_epochs,
            lr=args.lr
        )
        layer_pruning_ratios.append(pruning_ratio)

        # Compute MI statistics
        print(f"\n  Computing MI for layer {layer_idx + 1}...")
        avg_mi, all_mi, output_entropy = compute_layer_mi_stats(
            activations[layer_idx], output_probs, args.neurons_per_layer
        )
        layer_avg_mis.append(avg_mi)
        layer_entropies.append(output_entropy)

        # Store layer results
        results['layer_results'][layer_idx] = {
            'min_neurons': min_neurons,
            'pruning_ratio': pruning_ratio,
            'avg_mi': avg_mi,
            'output_entropy': output_entropy,
            'all_mi': {str(k): v for k, v in all_mi.items()}  # Convert tuple keys to strings
        }

    # Store summary arrays
    results['layer_pruning_ratios'] = np.array(layer_pruning_ratios)
    results['layer_avg_mis'] = np.array(layer_avg_mis)
    results['layer_entropies'] = np.array(layer_entropies)

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Pruning Ratio':<15} {'Avg MI':<15} {'Entropy':<15}")
    print("-" * 55)
    for i in range(args.n_layers):
        print(f"{i+1:<10} {layer_pruning_ratios[i]:<15.4f} {layer_avg_mis[i]:<15.4f} {layer_entropies[i]:<15.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/exp9_seed{args.seed}_results.npz"

    # Flatten layer_results for npz saving
    flat_results = {
        'seed': results['seed'],
        'n_layers': results['n_layers'],
        'neurons_per_layer': results['neurons_per_layer'],
        'train_acc': results['train_acc'],
        'test_acc': results['test_acc'],
        'epochs': results['epochs'],
        'layer_pruning_ratios': results['layer_pruning_ratios'],
        'layer_avg_mis': results['layer_avg_mis'],
        'layer_entropies': results['layer_entropies'],
    }

    # Add per-layer details
    for layer_idx, layer_data in results['layer_results'].items():
        flat_results[f'layer{layer_idx}_min_neurons'] = layer_data['min_neurons']
        flat_results[f'layer{layer_idx}_pruning_ratio'] = layer_data['pruning_ratio']
        flat_results[f'layer{layer_idx}_avg_mi'] = layer_data['avg_mi']
        flat_results[f'layer{layer_idx}_output_entropy'] = layer_data['output_entropy']

    np.savez(save_path, **flat_results)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
