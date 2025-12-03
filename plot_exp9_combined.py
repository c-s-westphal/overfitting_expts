#!/usr/bin/env python3
"""
Generate publication-quality combined plot for Experiment 9.
Shows assumption validity across all dropout rates with professional styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from math import comb
import os


# Set Times New Roman font globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman well


# Publication-quality color scheme (matching combined_exp1_exp2)
DROPOUT_COLORS = {
    0.0: '#3C5488',   # Blue
    0.1: '#00A087',   # Teal
    0.2: '#F39B7F',   # Salmon/peach
    0.3: '#E64B35',   # Red-orange
    0.4: '#8E44AD',   # Purple
    0.5: '#2C3E50',   # Dark blue-grey
}

DROPOUT_LABELS = {
    0.0: 'Dropout = 0.0',
    0.1: 'Dropout = 0.1',
    0.2: 'Dropout = 0.2',
    0.3: 'Dropout = 0.3',
    0.4: 'Dropout = 0.4',
    0.5: 'Dropout = 0.5',
}


def assumption_true(q_j, n_j, gamma_i):
    """
    Test whether the assumption holds for a given deeper layer j and shallower layer i.
    Returns True if: sum_{r=q_j}^{n_j} C(n_j, r) >= (2^{n_j} - 2) / gamma_i
    """
    lhs = sum(comb(n_j, r) for r in range(q_j, n_j + 1))
    rhs = (2**n_j - 2) / gamma_i
    return lhs >= rhs


def get_boundary_gamma(q, n):
    """
    Compute gamma value at the boundary for a given q.
    At boundary: sum_{r=q}^{n} C(n, r) = (2^n - 2) / gamma
    So: gamma = (2^n - 2) / sum_{r=q}^{n} C(n, r)
    """
    lhs = sum(comb(n, r) for r in range(q, n + 1))
    if lhs > 0:
        return (2**n - 2) / lhs
    return np.nan


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

            # Skip files without q_mi (MI-based metric)
            if 'layer0_q_mi' not in data.keys():
                print(f"  Skipping {filename} - no q_mi data")
                continue

            for layer_idx in range(n_layers):
                # Use q_mi (MI-based metric)
                q = int(data[f'layer{layer_idx}_q_mi'])

                all_data.append({
                    'seed': seed,
                    'dataset': dataset,
                    'n_layers': n_layers,
                    'dropout': dropout,
                    'layer': layer_idx,
                    'n': neurons_per_layer,
                    'q': q,
                    'gamma': float(data[f'layer{layer_idx}_gamma']),
                })

    return pd.DataFrame(all_data)


def plot_exp9_combined():
    """Create publication-quality combined plot."""

    # Set random seed for reproducible jitter
    np.random.seed(42)

    # Load data
    df = load_exp9_results()

    if df.empty:
        print("No data found!")
        return

    n_neurons = df['n'].iloc[0]  # 5 neurons

    # Find gamma range in data and set y-axis limits with 5% padding
    max_gamma = df['gamma'].max()
    min_gamma = df['gamma'].min()
    y_max = max_gamma * 1.05
    y_min = min_gamma * 0.95

    # Compute overall satisfaction rate for title
    total_pairs = 0
    satisfied_pairs = 0
    for dropout in df['dropout'].unique():
        dropout_df = df[df['dropout'] == dropout]
        for seed in dropout_df['seed'].unique():
            seed_df = dropout_df[dropout_df['seed'] == seed]
            layers = sorted(seed_df['layer'].unique())
            for i in layers:
                for j in layers:
                    if j > i:
                        q_j = seed_df.loc[seed_df['layer'] == j, 'q'].values[0]
                        n_j = seed_df.loc[seed_df['layer'] == j, 'n'].values[0]
                        gamma_i = seed_df.loc[seed_df['layer'] == i, 'gamma'].values[0]
                        total_pairs += 1
                        if assumption_true(int(q_j), int(n_j), gamma_i):
                            satisfied_pairs += 1
    overall_rate = 100 * satisfied_pairs / total_pairs if total_pairs > 0 else 0

    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Create grids for blue/red regions (extend slightly beyond visible range for full coverage)
    q_range_fine = np.linspace(0.5, n_neurons + 0.5, 500)
    gamma_range = np.linspace(y_min - 0.1, y_max + 0.1, 500)
    Q, G = np.meshgrid(q_range_fine, gamma_range)

    # Evaluate assumption on grid (using floor of q for discrete check)
    Z = np.zeros_like(Q, dtype=float)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            q_int = int(np.floor(Q[i, j]))
            Z[i, j] = 1.0 if assumption_true(q_int, n_neurons, G[i, j]) else 0.0

    # Plot blue/red regions with softer colors
    ax.contourf(Q, G, Z, levels=[-0.5, 0.5, 1.5],
                colors=['#FFCCCC', '#CCE5FF'], alpha=0.5)

    # Compute exact boundary curve as step function
    # The boundary is where gamma = (2^n - 2) / sum_{r=q}^{n} C(n, r)
    # This creates steps at each integer q value
    boundary_q = []
    boundary_gamma = []

    for q in range(1, n_neurons + 1):  # Start from 1 since x-range is 1-5
        gamma_at_q = get_boundary_gamma(q, n_neurons)
        if not np.isnan(gamma_at_q):
            # Add horizontal segment from previous q
            if boundary_gamma:
                boundary_q.append(q)
                boundary_gamma.append(boundary_gamma[-1])
            # Add vertical segment
            boundary_q.append(q)
            # Cap at y_max if boundary exceeds visible range
            boundary_gamma.append(min(gamma_at_q, y_max))

    # Extend to the right edge at the final (capped) gamma value
    if boundary_gamma:
        boundary_q.append(n_neurons)
        boundary_gamma.append(boundary_gamma[-1])

    # Plot the step-function boundary (dashed)
    ax.plot(boundary_q, boundary_gamma, color='black', linewidth=2.5,
            linestyle='--', zorder=10)

    # Collect all layer pairs for each dropout rate
    dropout_rates = sorted(df['dropout'].unique())

    # Track plotted points for legend
    legend_handles = []

    # Store all gamma values for determining text position
    all_gammas = []

    for dropout in dropout_rates:
        dropout_df = df[df['dropout'] == dropout]
        color = DROPOUT_COLORS.get(dropout, '#888888')
        label = DROPOUT_LABELS.get(dropout, f'Dropout = {dropout}')

        satisfied_q = []
        satisfied_gamma = []
        unsatisfied_q = []
        unsatisfied_gamma = []

        # Get all layer pairs (i, j) where j > i
        for seed in dropout_df['seed'].unique():
            seed_df = dropout_df[dropout_df['seed'] == seed]
            layers = sorted(seed_df['layer'].unique())

            for i in layers:
                for j in layers:
                    if j > i:
                        q_j = seed_df.loc[seed_df['layer'] == j, 'q'].values[0]
                        n_j = seed_df.loc[seed_df['layer'] == j, 'n'].values[0]
                        gamma_i = seed_df.loc[seed_df['layer'] == i, 'gamma'].values[0]

                        all_gammas.append(gamma_i)

                        if assumption_true(int(q_j), int(n_j), gamma_i):
                            satisfied_q.append(q_j)
                            satisfied_gamma.append(gamma_i)
                        else:
                            unsatisfied_q.append(q_j)
                            unsatisfied_gamma.append(gamma_i)

        # Add small jitter to avoid overlapping points
        jitter_scale = 0.02

        # Plot satisfied points (filled circles)
        if satisfied_q:
            jitter_q = np.array(satisfied_q) + np.random.uniform(-jitter_scale, jitter_scale, len(satisfied_q))
            jitter_gamma = np.array(satisfied_gamma) + np.random.uniform(-jitter_scale*0.5, jitter_scale*0.5, len(satisfied_gamma))
            ax.scatter(jitter_q, jitter_gamma, s=80, c=color, marker='o',
                      edgecolors='white', linewidths=1.0, zorder=5, alpha=0.85)

        # Plot unsatisfied points (X markers)
        if unsatisfied_q:
            jitter_q = np.array(unsatisfied_q) + np.random.uniform(-jitter_scale, jitter_scale, len(unsatisfied_q))
            jitter_gamma = np.array(unsatisfied_gamma) + np.random.uniform(-jitter_scale*0.5, jitter_scale*0.5, len(unsatisfied_gamma))
            ax.scatter(jitter_q, jitter_gamma, s=80, c=color, marker='X',
                      edgecolors='white', linewidths=1.0, zorder=5, alpha=0.85)

        # Create legend entry
        legend_handles.append(plt.scatter([], [], s=80, c=color, marker='o',
                                          edgecolors='white', linewidths=1.0, label=label))

    # Add legend entries for satisfied/unsatisfied
    legend_handles.append(plt.scatter([], [], s=80, c='gray', marker='o',
                                      edgecolors='white', linewidths=1.0,
                                      label='Assumption Satisfied'))
    legend_handles.append(plt.scatter([], [], s=80, c='gray', marker='X',
                                      edgecolors='white', linewidths=1.0,
                                      label='Assumption Violated'))

    # Add boundary line to legend with formula (dashed)
    legend_handles.insert(0, plt.Line2D([0], [0], color='black', linewidth=2.5,
                                        linestyle='--',
                                        label=r'$\sum_{r=q}^{n}\binom{n}{r} = \frac{2^{n}-2}{\gamma}$'))

    # Styling
    ax.set_xlabel(r'$q$', fontsize=13)
    ax.set_ylabel(r'$\gamma$', fontsize=13)
    ax.set_title(f'Layerwise Pruning Assumption Validity ({overall_rate:.1f}% Satisfied)\n'
                 f'MNIST Binary, 5 Layers, 5 Neurons per Layer',
                fontsize=14, fontweight='bold', pad=10)

    ax.set_xlim(0.7, n_neurons + 0.3)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(1, n_neurons + 1))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend below the graph
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=9, frameon=True, fancybox=False, edgecolor='black', framealpha=1)

    # Publication styling
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', labelsize=11, direction='out')

    # Add text annotation for regions (positioned based on data range)
    ax.text(1.3, y_max * 0.98, 'Assumption\nSatisfied', fontsize=10, ha='center', va='top',
            color='#3366AA', fontweight='medium', alpha=0.8)
    ax.text(4.5, y_min * 1.02, 'Assumption\nViolated', fontsize=10, ha='center', va='bottom',
            color='#AA3333', fontweight='medium', alpha=0.8)

    plt.tight_layout()

    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/exp9_assumption_validity_combined.png', dpi=600,
                bbox_inches='tight', facecolor='white')
    plt.savefig('plots/exp9_assumption_validity_combined.pdf',
                bbox_inches='tight', facecolor='white')
    print("Saved to plots/exp9_assumption_validity_combined.png")
    print("Saved to plots/exp9_assumption_validity_combined.pdf")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    for dropout in dropout_rates:
        dropout_df = df[df['dropout'] == dropout]
        total_pairs = 0
        satisfied_pairs = 0

        for seed in dropout_df['seed'].unique():
            seed_df = dropout_df[dropout_df['seed'] == seed]
            layers = sorted(seed_df['layer'].unique())

            for i in layers:
                for j in layers:
                    if j > i:
                        q_j = seed_df.loc[seed_df['layer'] == j, 'q'].values[0]
                        n_j = seed_df.loc[seed_df['layer'] == j, 'n'].values[0]
                        gamma_i = seed_df.loc[seed_df['layer'] == i, 'gamma'].values[0]

                        total_pairs += 1
                        if assumption_true(int(q_j), int(n_j), gamma_i):
                            satisfied_pairs += 1

        rate = 100 * satisfied_pairs / total_pairs if total_pairs > 0 else 0
        print(f"Dropout {dropout}: {satisfied_pairs}/{total_pairs} pairs satisfied ({rate:.1f}%)")


if __name__ == "__main__":
    print("Generating combined Experiment 9 plot...")
    plot_exp9_combined()
    print("\nDone!")
