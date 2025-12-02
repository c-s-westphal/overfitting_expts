"""
Plot to visualize where the layerwise pruning-ratio with uniform attenuation
assumption holds or fails.

The theoretical assumption says that for each deeper layer j > i:
    sum_{r=q_j}^{n_j} C(n_j, r) >= (2^{n_j} - 1) / gamma_i

Blue region: assumption true
Red region: assumption false
Points: measured (q_j, gamma_i) pairs from experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb
import os


def assumption_true(q_j, n_j, gamma_i):
    """
    Test whether the assumption holds for a given deeper layer j and shallower layer i.

    Returns True if: sum_{r=q_j}^{n_j} C(n_j, r) >= (2^{n_j} - 1) / gamma_i
    """
    lhs = sum(comb(n_j, r) for r in range(q_j, n_j + 1))
    rhs = (2**n_j - 1) / gamma_i
    return lhs >= rhs


def load_exp9_results(results_dir='results/exp9', dataset_filter=None):
    """Load all exp9 results and aggregate across seeds.

    Args:
        results_dir: Directory containing result files
        dataset_filter: If specified, only load results for this dataset
                       ('mnist_binary' or 'mnist_full')
    """
    all_data = []

    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('.npz'):
            filepath = os.path.join(results_dir, filename)
            data = np.load(filepath, allow_pickle=True)

            seed = int(data['seed'])
            n_layers = int(data['n_layers'])
            neurons_per_layer = int(data['neurons_per_layer'])

            # Get dataset info (handle old files without dataset field)
            if 'dataset' in data.keys():
                dataset = str(data['dataset'])
            else:
                dataset = 'mnist_binary'  # Default for old files

            # Filter by dataset if specified
            if dataset_filter is not None and dataset != dataset_filter:
                continue

            num_classes = int(data['num_classes']) if 'num_classes' in data.keys() else 2

            for layer_idx in range(n_layers):
                all_data.append({
                    'seed': seed,
                    'dataset': dataset,
                    'num_classes': num_classes,
                    'layer': layer_idx,
                    'n': neurons_per_layer,
                    'q': int(data[f'layer{layer_idx}_q']),
                    'gamma': float(data[f'layer{layer_idx}_gamma']),
                    'avg_mi': float(data[f'layer{layer_idx}_avg_mi']),
                    'output_entropy': float(data[f'layer{layer_idx}_output_entropy']),
                })

    return pd.DataFrame(all_data)


def plot_assumption_validity(df, output_path=None):
    """Create the assumption validity plot."""

    if df.empty:
        print("No data to plot!")
        return None

    # Get dataset info for title
    dataset = df['dataset'].iloc[0] if 'dataset' in df.columns else 'mnist_binary'
    dataset_label = "MNIST Full (10 classes)" if dataset == 'mnist_full' else "MNIST Binary (0 vs 1)"

    # Set output path based on dataset
    if output_path is None:
        output_path = f'plots/exp9_{dataset}_assumption_validity.png'

    # Get n_j (assuming all layers have same number of neurons)
    n_j = df['n'].iloc[0]

    # Create grids for visualization
    q_range = np.arange(0, n_j + 1, 1)
    gamma_range = np.linspace(0.5, 10.0, 300)
    Q, G = np.meshgrid(q_range, gamma_range)

    # Evaluate the assumption on the grid
    Z = np.zeros_like(Q, dtype=float)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            Z[i, j] = 1.0 if assumption_true(int(Q[i, j]), n_j, G[i, j]) else 0.0

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot blue/red regions
    plt.contourf(Q, G, Z, levels=[-0.5, 0.5, 1.5],
                 colors=['#ff6b6b', '#4dabf7'], alpha=0.4)

    # Compute boundary curve: gamma = (2^n - 1) / sum_{r=q}^{n} C(n, r)
    gamma_boundary = []
    for qj in q_range:
        lhs = sum(comb(n_j, r) for r in range(qj, n_j + 1))
        gamma_boundary.append((2**n_j - 1) / lhs if lhs > 0 else np.nan)

    plt.plot(q_range, gamma_boundary, 'k--', linewidth=2, label='Assumption Boundary')

    # Get all valid layer pairs (i, j) where j > i, across all seeds
    pairs_data = []
    for seed in df['seed'].unique():
        seed_df = df[df['seed'] == seed]
        layers = sorted(seed_df['layer'].unique())

        for i in layers:
            for j in layers:
                if j > i:
                    q_j = seed_df.loc[seed_df['layer'] == j, 'q'].values[0]
                    n_j_val = seed_df.loc[seed_df['layer'] == j, 'n'].values[0]
                    gamma_i = seed_df.loc[seed_df['layer'] == i, 'gamma'].values[0]

                    truth = assumption_true(int(q_j), int(n_j_val), gamma_i)
                    pairs_data.append({
                        'seed': seed,
                        'i': i,
                        'j': j,
                        'q_j': q_j,
                        'gamma_i': gamma_i,
                        'truth': truth
                    })

    pairs_df = pd.DataFrame(pairs_data)

    # Plot points
    for _, row in pairs_df.iterrows():
        if row['truth']:
            plt.scatter(row['q_j'], row['gamma_i'], s=100, color='darkblue',
                       edgecolor='white', marker='o', zorder=3, linewidths=1.5)
        else:
            plt.scatter(row['q_j'], row['gamma_i'], s=100, color='darkred',
                       marker='x', zorder=3, linewidths=2)

    # Add jitter for visibility if points overlap
    # Count unique (q_j, gamma_i) pairs
    unique_points = pairs_df.groupby(['q_j', 'gamma_i']).size().reset_index(name='count')

    # Compute and display empirical satisfaction rate
    total = len(pairs_df)
    true_count = pairs_df['truth'].sum()
    satisfaction_rate = 100 * true_count / total if total > 0 else 0

    # Add text annotation
    plt.text(0.02, 0.98, f"Assumption satisfied: {true_count}/{total} ({satisfaction_rate:.1f}%)",
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel(r"$q_j$ (min subset size with MI $\geq$ H(Y) at layer j)", fontsize=12)
    plt.ylabel(r"$\gamma_i$ (H(Y) / avg MI at layer i)", fontsize=12)
    plt.title(f"Layerwise Pruning Assumption Validity - {dataset_label}\n" +
              r"$\sum_{r=q_j}^{n_j} \binom{n_j}{r} \geq \frac{2^{n_j}-1}{\gamma_i}$", fontsize=12)
    plt.xlim(-0.5, n_j + 0.5)
    plt.xticks(q_range)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved to {output_path}")

    # Print summary
    print(f"\nAssumption satisfied in {true_count}/{total} pairs ({satisfaction_rate:.1f}%)")
    print("\nPairs breakdown:")
    print(pairs_df.to_string(index=False))

    plt.show()

    return pairs_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot exp9 assumption validity')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mnist_binary', 'mnist_full'],
                        help='Filter by dataset (default: plot all)')
    parser.add_argument('--results_dir', type=str, default='results/exp9',
                        help='Directory containing results')
    args = parser.parse_args()

    # Load data
    df = load_exp9_results(results_dir=args.results_dir, dataset_filter=args.dataset)

    if df.empty:
        print(f"No results found in {args.results_dir}" +
              (f" for dataset {args.dataset}" if args.dataset else ""))
    else:
        print("Loaded data:")
        print(df.to_string(index=False))
        print()

        # Create plot
        pairs_df = plot_assumption_validity(df)
