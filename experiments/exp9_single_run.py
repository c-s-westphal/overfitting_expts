"""
Experiment 9: MI-based pruning analysis for binary MNIST classification.

This experiment:
1. Trains a small MLP on MNIST binary (0 vs 1) with dropout before output layer
2. For each layer, computes:
   - MI between individual neurons and output (class labels) using LNC-KSG
   - q = number of neurons with MI > 0 (neurons that cannot be pruned)
   - Pruning ratio = (n - q) / n (fraction of neurons with MI = 0)
   - MI between all neuron subsets and output
   - Discrete entropy H(Y) and gamma = H(Y) / avg_MI
3. Stores per-layer: q, pruning_ratio, neuron_mis, avg_mi, H(Y), gamma
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
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_mnist_binary_dataloaders


class MLP_Binary(nn.Module):
    """
    Small MLP for binary MNIST classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - N hidden layers with M neurons each
    - Dropout before output layer
    - Output: 2 classes (digits 0 and 1)
    """
    def __init__(self, n_layers=5, neurons_per_layer=5, num_classes=2, dropout=0.5):
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

        # Dropout before output layer
        self.dropout = nn.Dropout(p=dropout)

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
        x = self.dropout(x)
        return self.output_layer(x)

    def forward_with_activations(self, x):
        """Forward pass returning activations at each hidden layer."""
        x = x.view(x.size(0), -1)
        activations = []
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
            activations.append(x)
        x = self.dropout(x)
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


def collect_activations_and_labels(model, dataloader, device):
    """
    Collect all hidden layer activations and class labels on training set.

    Returns:
        activations: List of arrays, one per layer, shape (N, neurons_per_layer)
        labels: Array of class labels (0 or 1), shape (N,)
    """
    model.eval()
    all_activations = [[] for _ in range(model.n_layers)]
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            output, acts = model.forward_with_activations(inputs)

            for i, act in enumerate(acts):
                all_activations[i].append(act.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate
    activations = [np.concatenate(a, axis=0) for a in all_activations]
    labels = np.concatenate(all_labels, axis=0)

    return activations, labels


def add_jitter(data, jitter_scale=1e-10):
    """
    Add small jitter to data to handle identical points.
    """
    data = data.copy().astype(np.float64)
    noise = np.random.randn(*data.shape) * jitter_scale
    return data + noise


def check_dead_neurons(X):
    """Check which neurons have zero variance (dead neurons)."""
    if X.ndim == 1:
        return np.std(X) < 1e-10
    return np.std(X, axis=0) < 1e-10


def compute_discrete_entropy(labels):
    """
    Compute discrete entropy H(Y) for class labels in bits.

    H(Y) = -sum_y p(y) log2 p(y)

    Args:
        labels: Array of discrete class labels

    Returns:
        Entropy in bits
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def compute_ksg_mi(X, Y, k=5):
    """
    Compute MI using the KSG (Kraskov-Stögbauer-Grassberger) estimator.

    Based on: Kraskov et al., "Estimating Mutual Information", PRE 2004.

    Uses the first KSG estimator (Algorithm 1).

    Args:
        X: Array of shape (N, d_x) - continuous features
        Y: Array of shape (N,) or (N, d_y) - continuous target
        k: Number of nearest neighbors (default: 5)

    Returns:
        MI estimate in nats
    """
    # Ensure proper shapes
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples = X.shape[0]
    d_x = X.shape[1]
    d_y = Y.shape[1]

    # Check for dead neurons
    dead_mask = check_dead_neurons(X)
    if np.all(dead_mask):
        return 0.0

    # Add small jitter to avoid identical points
    X = add_jitter(X)
    Y = add_jitter(Y)

    # Normalize to unit variance for better distance computation
    X_std = np.std(X, axis=0, keepdims=True)
    X_std[X_std < 1e-10] = 1.0
    X_norm = X / X_std

    Y_std = np.std(Y, axis=0, keepdims=True)
    Y_std[Y_std < 1e-10] = 1.0
    Y_norm = Y / Y_std

    # Joint space
    XY = np.hstack([X_norm, Y_norm])

    # Build KD-trees
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X_norm)
    tree_y = cKDTree(Y_norm)

    # For each point, find distance to k-th neighbor in joint space
    # query k+1 because the point itself is included
    distances_xy, _ = tree_xy.query(XY, k=k+1, p=np.inf)  # Chebyshev distance
    eps = distances_xy[:, -1]  # k-th neighbor distance

    # Count points within eps in marginal spaces
    n_x = np.zeros(n_samples)
    n_y = np.zeros(n_samples)

    for i in range(n_samples):
        # Count points strictly within eps (not including boundary)
        n_x[i] = len(tree_x.query_ball_point(X_norm[i], eps[i] - 1e-15, p=np.inf)) - 1
        n_y[i] = len(tree_y.query_ball_point(Y_norm[i], eps[i] - 1e-15, p=np.inf)) - 1

    # KSG estimator formula
    # I(X;Y) = psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>
    mi = digamma(k) + digamma(n_samples) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))

    return max(0.0, mi)  # MI is non-negative


def compute_lnc_ksg_mi(X, Y, k=5, alpha=0.25):
    """
    Compute MI using LNC-KSG (Local Non-uniform Correction KSG) estimator.

    Based on: Gao et al., "Demystifying Fixed k-Nearest Neighbor Information Estimators"

    The LNC correction accounts for non-uniform density in local neighborhoods
    using PCA-based volume correction.

    Args:
        X: Array of shape (N, d_x) - continuous features
        Y: Array of shape (N,) or (N, d_y) - continuous target
        k: Number of nearest neighbors (default: 5)
        alpha: Fraction of neighbors to use for LNC (default: 0.25)

    Returns:
        MI estimate in nats
    """
    # Ensure proper shapes
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples = X.shape[0]
    d_x = X.shape[1]
    d_y = Y.shape[1]
    d_xy = d_x + d_y

    # Check for dead neurons
    dead_mask = check_dead_neurons(X)
    if np.all(dead_mask):
        return 0.0

    # Add small jitter
    X = add_jitter(X)
    Y = add_jitter(Y)

    # Normalize to unit variance
    X_std = np.std(X, axis=0, keepdims=True)
    X_std[X_std < 1e-10] = 1.0
    X_norm = X / X_std

    Y_std = np.std(Y, axis=0, keepdims=True)
    Y_std[Y_std < 1e-10] = 1.0
    Y_norm = Y / Y_std

    XY = np.hstack([X_norm, Y_norm])

    # Build KD-trees
    tree_xy = cKDTree(XY)
    tree_x = cKDTree(X_norm)
    tree_y = cKDTree(Y_norm)

    # Number of neighbors for LNC
    k_lnc = max(1, int(alpha * k))

    # For each point, find k-th neighbor distance in joint space
    distances_xy, indices_xy = tree_xy.query(XY, k=k+1, p=np.inf)
    eps = distances_xy[:, -1]

    # Count points in marginal spaces and compute LNC correction
    n_x = np.zeros(n_samples)
    n_y = np.zeros(n_samples)
    lnc_correction = np.zeros(n_samples)

    for i in range(n_samples):
        # Points within eps in marginals
        n_x[i] = len(tree_x.query_ball_point(X_norm[i], eps[i] - 1e-15, p=np.inf)) - 1
        n_y[i] = len(tree_y.query_ball_point(Y_norm[i], eps[i] - 1e-15, p=np.inf)) - 1

        # LNC correction: estimate local dimensionality using PCA
        if k_lnc >= d_xy:
            neighbor_indices = indices_xy[i, 1:k+1]  # Exclude self
            local_points = XY[neighbor_indices] - XY[i]

            try:
                # SVD to estimate local dimensionality
                _, s, _ = np.linalg.svd(local_points, full_matrices=False)
                # Ratio of explained variance
                var_ratio = s ** 2 / (np.sum(s ** 2) + 1e-15)
                # Effective dimensionality (number of significant components)
                eff_dim = np.sum(var_ratio > 0.01)
                # Correction based on difference from full dimensionality
                lnc_correction[i] = (d_xy - eff_dim) * np.log(2) / d_xy
            except:
                lnc_correction[i] = 0.0

    # KSG with LNC correction
    mi = (digamma(k) + digamma(n_samples)
          - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
          + np.mean(lnc_correction))

    # Convert from nats to bits
    mi_bits = mi / np.log(2)

    return max(0.0, mi_bits)


def compute_layer_mi_stats(activations, labels, neurons_per_layer=4, k=5, mi_threshold=1e-6):
    """
    Compute MI statistics for all subsets of neurons in a layer.

    Uses LNC-KSG estimator for MI (in bits) and discrete entropy for H(Y).
    Also computes q (number of neurons with MI > 0) and pruning ratio.

    Args:
        activations: Array of shape (N, neurons_per_layer) - layer activations
        labels: Array of shape (N,) - discrete class labels (0 or 1)
        neurons_per_layer: Number of neurons in the layer
        k: Number of neighbors for KSG estimator
        mi_threshold: Threshold below which MI is considered 0

    Returns:
        avg_mi: Average MI across all subsets (in bits)
        all_mi: Dict mapping subset tuple to MI value
        output_entropy: Discrete entropy H(Y) in bits
        gamma: H(Y) / avg_mi (attenuation factor)
        q: Number of neurons with MI > 0 (neurons to keep)
        pruning_ratio: Fraction of neurons that can be pruned (MI = 0)
        neuron_mis: List of individual neuron MIs
    """
    # Compute discrete output entropy H(Y) in bits
    output_entropy = compute_discrete_entropy(labels)
    print(f"    Output entropy H(Y): {output_entropy:.4f} bits")

    all_mi = {}
    mi_values = []
    neuron_mis = []  # MI for individual neurons (subsets of size 1)

    # Generate all subsets of size 1 to neurons_per_layer-1
    neuron_indices = list(range(neurons_per_layer))

    for subset_size in range(1, neurons_per_layer):  # 1, 2, 3
        for subset in combinations(neuron_indices, subset_size):
            # Extract activations for this subset
            subset_activations = activations[:, list(subset)]

            # Compute MI using LNC-KSG (returns bits)
            mi = compute_lnc_ksg_mi(subset_activations, labels.reshape(-1, 1), k=k)
            all_mi[subset] = mi
            mi_values.append(mi)

            # Track individual neuron MIs
            if subset_size == 1:
                neuron_mis.append(mi)
                print(f"    Neuron {subset[0]}: MI = {mi:.4f} bits")
            else:
                print(f"    Subset {subset}: MI = {mi:.4f} bits")

    avg_mi = np.nanmean(mi_values)
    print(f"    Average MI across all subsets: {avg_mi:.4f} bits")

    # Compute q = number of neurons with MI > threshold
    neuron_mis = np.array(neuron_mis)
    q = int(np.sum(neuron_mis > mi_threshold))
    pruning_ratio = (neurons_per_layer - q) / neurons_per_layer

    print(f"    Neurons with MI > 0: {q}/{neurons_per_layer}")
    print(f"    Pruning ratio: {pruning_ratio:.4f}")

    # Compute gamma = H(Y) / avg_MI
    if avg_mi > 1e-10:
        gamma = output_entropy / avg_mi
    else:
        gamma = np.inf
    print(f"    Gamma (H(Y) / avg_MI): {gamma:.4f}")

    return avg_mi, all_mi, output_entropy, gamma, q, pruning_ratio, neuron_mis


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 9: Pruning ratio and MI analysis for binary MNIST'
    )
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('--neurons_per_layer', type=int, default=4,
                        help='Neurons per hidden layer (default: 4)')
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
    parser.add_argument('--target_train_acc', type=float, default=100.0,
                        help='Target train accuracy')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate before output layer (default: 0.5)')

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
    print(f"Dropout:           {args.dropout}")
    print(f"Target Train Acc:  {args.target_train_acc}%")
    print(f"{'='*80}\n")

    # Create model
    model = MLP_Binary(
        n_layers=args.n_layers,
        neurons_per_layer=args.neurons_per_layer,
        num_classes=2,
        dropout=args.dropout
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

    # Collect activations and class labels
    print(f"\n{'='*80}")
    print("Collecting activations and class labels...")
    print(f"{'='*80}")

    activations, labels = collect_activations_and_labels(model, trainloader, args.device)
    print(f"Collected {len(labels)} samples")
    print(f"Class distribution: {np.bincount(labels)}")
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
    print("Per-layer Analysis: MI-based Pruning and Statistics (using LNC-KSG)")
    print(f"{'='*80}")

    layer_pruning_ratios = []
    layer_avg_mis = []
    layer_entropies = []
    layer_gammas = []
    layer_qs = []
    layer_neuron_mis = []

    for layer_idx in range(args.n_layers):
        print(f"\n--- Layer {layer_idx + 1} ---")

        # Compute MI statistics using LNC-KSG
        # q = number of neurons with MI > 0 (neurons that cannot be pruned)
        # pruning_ratio = fraction of neurons with MI = 0 (can be pruned)
        avg_mi, all_mi, output_entropy, gamma, q, pruning_ratio, neuron_mis = compute_layer_mi_stats(
            activations[layer_idx], labels, args.neurons_per_layer
        )
        layer_avg_mis.append(avg_mi)
        layer_entropies.append(output_entropy)
        layer_gammas.append(gamma)
        layer_qs.append(q)
        layer_pruning_ratios.append(pruning_ratio)
        layer_neuron_mis.append(neuron_mis)

        # Store layer results
        results['layer_results'][layer_idx] = {
            'q': q,
            'pruning_ratio': pruning_ratio,
            'avg_mi': avg_mi,
            'output_entropy': output_entropy,
            'gamma': gamma,
            'neuron_mis': neuron_mis.tolist(),
            'all_mi': {str(k): v for k, v in all_mi.items()}  # Convert tuple keys to strings
        }

    # Store summary arrays
    results['layer_pruning_ratios'] = np.array(layer_pruning_ratios)
    results['layer_avg_mis'] = np.array(layer_avg_mis)
    results['layer_entropies'] = np.array(layer_entropies)
    results['layer_gammas'] = np.array(layer_gammas)
    results['layer_qs'] = np.array(layer_qs)

    # Print summary
    print(f"\n{'='*80}")
    print("Summary (MI in bits, H(Y) discrete)")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'q':<8} {'Prune Ratio':<12} {'Avg MI':<12} {'H(Y)':<12} {'Gamma':<12}")
    print("-" * 64)
    for i in range(args.n_layers):
        print(f"{i+1:<8} {layer_qs[i]:<8} {layer_pruning_ratios[i]:<12.4f} {layer_avg_mis[i]:<12.4f} {layer_entropies[i]:<12.4f} {layer_gammas[i]:<12.4f}")

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
        'dropout': args.dropout,
        'layer_pruning_ratios': results['layer_pruning_ratios'],
        'layer_avg_mis': results['layer_avg_mis'],
        'layer_entropies': results['layer_entropies'],
        'layer_gammas': results['layer_gammas'],
        'layer_qs': results['layer_qs'],
    }

    # Add per-layer details
    for layer_idx, layer_data in results['layer_results'].items():
        flat_results[f'layer{layer_idx}_q'] = layer_data['q']
        flat_results[f'layer{layer_idx}_pruning_ratio'] = layer_data['pruning_ratio']
        flat_results[f'layer{layer_idx}_avg_mi'] = layer_data['avg_mi']
        flat_results[f'layer{layer_idx}_output_entropy'] = layer_data['output_entropy']
        flat_results[f'layer{layer_idx}_gamma'] = layer_data['gamma']
        flat_results[f'layer{layer_idx}_neuron_mis'] = np.array(layer_data['neuron_mis'])

    np.savez(save_path, **flat_results)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
