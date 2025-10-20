#!/usr/bin/env python3
"""
Plot Experiment 5 MNIST: MI difference vs Generalization Gap
Scatter plot with points grouped by number of layers.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict

def load_mnist_results(results_dir='results/exp5'):
    """Load all MNIST results and organize by layer count."""

    # Dictionary to store results for each layer count
    data_by_layers = defaultdict(lambda: {
        'ln10_minus_mi_masked': [],
        'gen_gaps': [],
        'seeds': []
    })

    # Load all MNIST result files
    pattern = os.path.join(results_dir, 'mlp_mnist_layers*_seed*_results.npz')
    files = glob.glob(pattern)

    print(f"Found {len(files)} MNIST result files")

    for fpath in files:
        # Extract layer count and seed from filename
        fname = os.path.basename(fpath)
        # Pattern: mlp_mnist_layers{n}_seed{s}_results.npz
        parts = fname.replace('.npz', '').replace('mlp_mnist_', '').split('_')
        n_layers = int(parts[0].replace('layers', ''))
        seed = int(parts[1].replace('seed', ''))

        # Load the data
        data = np.load(fpath, allow_pickle=True)

        # Extract metrics: ln(10) - subset_mi
        subset_mi = float(data['final_mean_mi_masked'])
        ln10_minus_mi_masked = np.log(10) - subset_mi
        gen_gap = float(data['final_gen_gap'])

        # Store in dictionary
        data_by_layers[n_layers]['ln10_minus_mi_masked'].append(ln10_minus_mi_masked)
        data_by_layers[n_layers]['gen_gaps'].append(gen_gap)
        data_by_layers[n_layers]['seeds'].append(seed)

    return data_by_layers

def plot_mi_diff_vs_gen_gap_by_layers(data_by_layers):
    """Create scatter plot of ln(10) - subset MI vs generalization gap, colored by layer count."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get layer counts and sort them
    layer_counts = sorted(data_by_layers.keys())

    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_counts)))

    # Plot each layer count with a different color
    for i, n_layers in enumerate(layer_counts):
        gen_gaps = data_by_layers[n_layers]['gen_gaps']
        ln10_minus_mi = data_by_layers[n_layers]['ln10_minus_mi_masked']

        ax.scatter(gen_gaps, ln10_minus_mi,
                  c=[colors[i]],
                  label=f'{n_layers} layer{"s" if n_layers > 1 else ""}',
                  s=100,
                  alpha=0.7,
                  edgecolors='black',
                  linewidths=0.5)

    # Labels and title
    ax.set_xlabel('Generalization Gap (%)', fontsize=14)
    ax.set_ylabel('ln(10) - Subset MI (bits)', fontsize=14)
    ax.set_title('ln(10) - Subset MI vs Generalization Gap (MNIST)\nGrouped by Network Depth', fontsize=16)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)

    # Compute overall correlation
    all_gen_gaps = []
    all_ln10_minus_mi = []
    for n_layers in layer_counts:
        all_gen_gaps.extend(data_by_layers[n_layers]['gen_gaps'])
        all_ln10_minus_mi.extend(data_by_layers[n_layers]['ln10_minus_mi_masked'])

    correlation = np.corrcoef(all_gen_gaps, all_ln10_minus_mi)[0, 1]

    # Add correlation text
    ax.text(0.05, 0.95, f'Overall Correlation: {correlation:.3f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_mnist_ln10_minus_subset_mi_vs_gen_gap_by_layers.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot saved: plots/experiment_5_mnist_ln10_minus_subset_mi_vs_gen_gap_by_layers.png")

def print_summary_statistics(data_by_layers):
    """Print summary statistics for each layer count."""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Layers':<8} {'ln(10)-Subset MI (mean±std)':<35} {'Gen Gap (mean±std)':<25} {'N':<5}")
    print("-"*80)

    layer_counts = sorted(data_by_layers.keys())

    for n_layers in layer_counts:
        ln10_minus_mi = np.array(data_by_layers[n_layers]['ln10_minus_mi_masked'])
        gen_gaps = np.array(data_by_layers[n_layers]['gen_gaps'])

        mi_mean = np.mean(ln10_minus_mi)
        mi_std = np.std(ln10_minus_mi)
        gap_mean = np.mean(gen_gaps)
        gap_std = np.std(gen_gaps)
        n = len(ln10_minus_mi)

        print(f"{n_layers:<8} {mi_mean:.4f} ± {mi_std:.4f}{'':<21} {gap_mean:.2f} ± {gap_std:.2f}%{'':<11} {n:<5}")

    print("="*80)

def main():
    print("Loading MNIST Experiment 5 results...")
    data_by_layers = load_mnist_results()

    if not data_by_layers:
        print("No MNIST results found!")
        return

    print("\nCreating plot...")
    plot_mi_diff_vs_gen_gap_by_layers(data_by_layers)

    print_summary_statistics(data_by_layers)

    print("\nDone!")

if __name__ == "__main__":
    main()
