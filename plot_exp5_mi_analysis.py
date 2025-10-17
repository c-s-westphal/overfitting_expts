#!/usr/bin/env python3
"""
Plot Experiment 5 MI analysis:
1. Number of layers vs MI difference (line with error bars)
2. Number of layers vs test accuracy (line with error bars)
3. MI difference vs test accuracy (scatter plot)
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict

def load_exp5_data(results_dir='results/exp5'):
    """Load all exp5 MLP results and organize by layer count."""

    # Dictionary to store results for each layer count
    data_by_layers = defaultdict(lambda: {
        'mi_diffs': [],
        'test_accs': [],
        'seeds': []
    })

    # Load all result files
    pattern = os.path.join(results_dir, 'mlp_layers*_seed*_results.npz')
    files = glob.glob(pattern)

    print(f"Found {len(files)} result files")

    for fpath in files:
        # Extract layer count and seed from filename
        fname = os.path.basename(fpath)
        # Pattern: mlp_layers{n}_seed{s}_results.npz
        parts = fname.replace('.npz', '').split('_')
        n_layers = int(parts[1].replace('layers', ''))
        seed = int(parts[2].replace('seed', ''))

        # Load the data
        data = np.load(fpath, allow_pickle=True)

        # Extract metrics
        mi_diff = float(data['final_mi_diff'])
        test_acc = float(data['final_test_acc'])

        # Store in dictionary
        data_by_layers[n_layers]['mi_diffs'].append(mi_diff)
        data_by_layers[n_layers]['test_accs'].append(test_acc)
        data_by_layers[n_layers]['seeds'].append(seed)

    return data_by_layers

def compute_statistics(data_by_layers):
    """Compute means and standard errors for each layer count."""

    layer_counts = sorted(data_by_layers.keys())

    mi_diff_means = []
    mi_diff_sems = []
    test_acc_means = []
    test_acc_sems = []

    for n_layers in layer_counts:
        mi_diffs = np.array(data_by_layers[n_layers]['mi_diffs'])
        test_accs = np.array(data_by_layers[n_layers]['test_accs'])

        # Compute mean and standard error
        mi_diff_means.append(np.mean(mi_diffs))
        mi_diff_sems.append(np.std(mi_diffs) / np.sqrt(len(mi_diffs)))

        test_acc_means.append(np.mean(test_accs))
        test_acc_sems.append(np.std(test_accs) / np.sqrt(len(test_accs)))

    return {
        'layer_counts': layer_counts,
        'mi_diff_means': mi_diff_means,
        'mi_diff_sems': mi_diff_sems,
        'test_acc_means': test_acc_means,
        'test_acc_sems': test_acc_sems
    }

def plot_layers_vs_mi_diff(stats):
    """Plot 1: Number of layers vs MI difference."""
    plt.figure(figsize=(10, 6))

    plt.errorbar(stats['layer_counts'], stats['mi_diff_means'],
                 yerr=stats['mi_diff_sems'],
                 marker='o', capsize=5, linewidth=2, markersize=8,
                 color='blue', label='MI Difference')

    plt.xlabel('Number of Hidden Layers', fontsize=14)
    plt.ylabel('MI Difference (bits)', fontsize=14)
    plt.title('Mutual Information Difference vs Network Depth', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_layers_vs_mi_diff.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot 1 saved: plots/experiment_5_layers_vs_mi_diff.png")

def plot_layers_vs_test_acc(stats):
    """Plot 2: Number of layers vs test accuracy."""
    plt.figure(figsize=(10, 6))

    plt.errorbar(stats['layer_counts'], stats['test_acc_means'],
                 yerr=stats['test_acc_sems'],
                 marker='o', capsize=5, linewidth=2, markersize=8,
                 color='red', label='Test Accuracy')

    plt.xlabel('Number of Hidden Layers', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title('Test Accuracy vs Network Depth', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_layers_vs_test_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot 2 saved: plots/experiment_5_layers_vs_test_acc.png")

def plot_mi_diff_vs_test_acc(data_by_layers):
    """Plot 3: MI difference vs test accuracy (scatter)."""
    plt.figure(figsize=(10, 6))

    # Collect all individual data points
    all_mi_diffs = []
    all_test_accs = []

    for n_layers in sorted(data_by_layers.keys()):
        all_mi_diffs.extend(data_by_layers[n_layers]['mi_diffs'])
        all_test_accs.extend(data_by_layers[n_layers]['test_accs'])

    # Create scatter plot
    plt.scatter(all_mi_diffs, all_test_accs, alpha=0.6, s=50, color='purple')

    plt.xlabel('MI Difference (bits)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.title('Test Accuracy vs Mutual Information Difference', fontsize=16)
    plt.grid(True, alpha=0.3)

    # Add correlation info
    correlation = np.corrcoef(all_mi_diffs, all_test_accs)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_mi_diff_vs_test_acc.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot 3 saved: plots/experiment_5_mi_diff_vs_test_acc.png")

def main():
    print("Loading Experiment 5 results...")
    data_by_layers = load_exp5_data()

    print("\nComputing statistics...")
    stats = compute_statistics(data_by_layers)

    print("\nGenerating plots...")
    plot_layers_vs_mi_diff(stats)
    plot_layers_vs_test_acc(stats)
    plot_mi_diff_vs_test_acc(data_by_layers)

    print("\nAll plots generated successfully!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for i, n_layers in enumerate(stats['layer_counts']):
        print(f"Layers {n_layers:2d}: "
              f"MI Diff = {stats['mi_diff_means'][i]:.4f} ± {stats['mi_diff_sems'][i]:.4f}, "
              f"Test Acc = {stats['test_acc_means'][i]:.2f} ± {stats['test_acc_sems'][i]:.2f}%")

if __name__ == "__main__":
    main()
