#!/usr/bin/env python3
"""
Plot gamma versus layer for Experiment 9.
Shows how gamma changes across layers for different dropout rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Publication-quality color scheme (matching combined plots)
DROPOUT_COLORS = {
    0.0: '#3C5488',   # Blue
    0.1: '#00A087',   # Teal
    0.2: '#F39B7F',   # Salmon/peach
    0.3: '#E64B35',   # Red-orange
}

DROPOUT_LABELS = {
    0.0: 'Dropout = 0.0',
    0.1: 'Dropout = 0.1',
    0.2: 'Dropout = 0.2',
    0.3: 'Dropout = 0.3',
}


def load_exp9_results(results_dir='results/exp9'):
    """Load all L5 N5 exp9 results."""
    all_data = []

    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('.npz') and 'L5_N5' in filename:
            filepath = os.path.join(results_dir, filename)
            data = np.load(filepath, allow_pickle=True)

            seed = int(data['seed'])
            n_layers = int(data['n_layers'])
            neurons_per_layer = int(data['neurons_per_layer'])

            # Get dropout rate
            if 'dropout_rate' in data.keys():
                dropout = float(data['dropout_rate'])
            elif 'dropout_base' in data.keys():
                dropout = float(data['dropout_base'])
            else:
                dropout = 0.0

            dataset = str(data['dataset']) if 'dataset' in data.keys() else 'mnist_binary'

            for layer_idx in range(n_layers):
                all_data.append({
                    'seed': seed,
                    'dataset': dataset,
                    'n_layers': n_layers,
                    'dropout': dropout,
                    'layer': layer_idx + 1,  # 1-indexed for display
                    'n': neurons_per_layer,
                    'gamma': float(data[f'layer{layer_idx}_gamma']),
                })

    return pd.DataFrame(all_data)


def plot_gamma_vs_layer():
    """Create publication-quality gamma vs layer plot with 4 subplots."""

    # Set random seed for reproducible jitter
    np.random.seed(42)

    # Load data
    df = load_exp9_results()

    if df.empty:
        print("No data found!")
        return

    n_layers = df['n_layers'].iloc[0]
    dropout_rates = sorted(df['dropout'].unique())

    # Create figure with 4 subplots side by side (independent y-axes)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)

    # Add small jitter to avoid overlapping points
    jitter_scale = 0.08

    for idx, dropout in enumerate(dropout_rates):
        ax = axes[idx]
        dropout_df = df[df['dropout'] == dropout]
        color = DROPOUT_COLORS.get(dropout, '#888888')
        label = DROPOUT_LABELS.get(dropout, f'Dropout = {dropout}')

        layers = dropout_df['layer'].values
        gammas = dropout_df['gamma'].values

        # Add jitter
        jitter_x = np.random.uniform(-jitter_scale, jitter_scale, len(layers))
        jitter_y = np.random.uniform(-jitter_scale * 0.5, jitter_scale * 0.5, len(gammas))

        # Plot points
        ax.scatter(layers + jitter_x, gammas + jitter_y,
                   s=60, c=color, marker='o',
                   edgecolors='white', linewidths=0.8,
                   zorder=5, alpha=0.85)

        # Compute and plot mean line
        mean_gamma = dropout_df.groupby('layer')['gamma'].mean()
        ax.plot(mean_gamma.index, mean_gamma.values,
                color=color, linewidth=2.5, linestyle='-', alpha=0.9, zorder=4)

        # Subplot title
        ax.set_title(f'Dropout = {dropout}', fontsize=12, fontweight='bold')

        # X-axis
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_xticks(range(1, n_layers + 1))

        # Y-axis (adapted to each subplot's data range)
        sub_min = dropout_df['gamma'].min()
        sub_max = dropout_df['gamma'].max()
        sub_padding = (sub_max - sub_min) * 0.15
        if sub_padding < 0.01:  # Minimum padding for very tight data
            sub_padding = 0.02
        ax.set_ylim(sub_min - sub_padding, sub_max + sub_padding)
        ax.set_ylabel(r'$\gamma$', fontsize=13)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Publication styling
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        ax.tick_params(axis='both', which='major', labelsize=10, direction='out')

    # Main title
    fig.suptitle('Gamma vs Layer by Dropout Rate\nMNIST Binary, 5 Layers, 5 Neurons per Layer',
                 fontsize=14, fontweight='bold', y=1.02)

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/exp9_gamma_vs_layer.png', dpi=600,
                bbox_inches='tight', facecolor='white')
    plt.savefig('plots/exp9_gamma_vs_layer.pdf',
                bbox_inches='tight', facecolor='white')
    print("Saved to plots/exp9_gamma_vs_layer.png")
    print("Saved to plots/exp9_gamma_vs_layer.pdf")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    for dropout in dropout_rates:
        dropout_df = df[df['dropout'] == dropout]
        mean_gamma = dropout_df['gamma'].mean()
        std_gamma = dropout_df['gamma'].std()
        print(f"Dropout {dropout}: mean γ = {mean_gamma:.3f} ± {std_gamma:.3f}")


if __name__ == "__main__":
    print("Generating gamma vs layer plot...")
    plot_gamma_vs_layer()
    print("\nDone!")
