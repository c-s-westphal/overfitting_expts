#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.results_utils import load_experiment_results, aggregate_results, load_exp1_results_from_per_seed, load_exp2_results_from_per_seed, load_exp3_results_from_per_seed, load_exp4_results_from_per_seed, load_exp5_results_from_per_seed


def plot_experiment_1():
    models = ['ResNet20', 'ResNet32', 'ResNet56', 'VGG11', 'VGG16', 'VGG19']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(12, 8))
    
    for model, color in zip(models, colors):
        try:
            # Prefer per-seed aggregation; fallback to pre-aggregated file if absent
            try:
                results = load_exp1_results_from_per_seed(model, results_dir='results/exp1')
            except FileNotFoundError:
                results = load_experiment_results('exp1', model)
            aggregated = aggregate_results(results)
            
            sizes = results['dataset_sizes']
            gaps_mean = aggregated['generalization_gaps_mean']
            gaps_std = aggregated['generalization_gaps_std']
            
            valid_sizes = []
            valid_gaps_mean = []
            valid_gaps_std = []
            
            for i, (size, gap_mean, gap_std) in enumerate(zip(sizes, gaps_mean, gaps_std)):
                if gap_mean is not None:
                    valid_sizes.append(size)
                    valid_gaps_mean.append(gap_mean)
                    valid_gaps_std.append(gap_std)
            
            if valid_sizes:
                plt.errorbar(valid_sizes, valid_gaps_mean, yerr=valid_gaps_std, 
                           label=model, color=color, marker='o', capsize=5)
        
        except FileNotFoundError:
            print(f"Results not found for {model}")
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Generalization Gap (%)')
    plt.title('Generalization Gap vs Training Set Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    os.makedirs('plots', exist_ok=True)
    #plt.savefig('plots/experiment_1_generalization_gap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Experiment 1 plot saved to plots/experiment_1_generalization_gap.png")


def plot_experiment_2():
    models = ['ResNet20', 'ResNet32', 'ResNet56', 'VGG11', 'VGG16', 'VGG19']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(12, 8))
    
    for model, color in zip(models, colors):
        try:
            # Prefer per-seed aggregation; fallback to pre-aggregated file if absent
            try:
                results = load_exp2_results_from_per_seed(model, results_dir='results/exp2')
            except FileNotFoundError:
                results = load_experiment_results('exp2', model)
            aggregated = aggregate_results(results)
            
            mi_values = results['mi_values']
            gaps_mean = aggregated['generalization_gaps_mean']
            gaps_std = aggregated['generalization_gaps_std']
            
            valid_mi = []
            valid_gaps_mean = []
            valid_gaps_std = []
            
            for i, (mi, gap_mean, gap_std) in enumerate(zip(mi_values, gaps_mean, gaps_std)):
                if gap_mean is not None:
                    valid_mi.append(mi)
                    valid_gaps_mean.append(gap_mean)
                    valid_gaps_std.append(gap_std)
            
            if valid_mi:
                plt.errorbar(valid_mi, valid_gaps_mean, yerr=valid_gaps_std, 
                           label=model, color=color, marker='o', capsize=5)
        
        except FileNotFoundError:
            print(f"Results not found for {model}")
    
    plt.xlabel('Mutual Information (bits)')
    plt.ylabel('Generalization Gap (%)')
    plt.title('Generalization Gap vs Mutual Information of Special Pixel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_2_mi_vs_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Experiment 2 plot saved to plots/experiment_2_mi_vs_gap.png")


def plot_experiment_3():
    """Plot Experiment 3: VGG Variable - Generalization Gap and Epochs to 100% vs Number of Layers."""
    models = [
        ('VGG11', 'blue'),
        ('VGG13', 'green'),
        ('VGG16', 'red'),
        ('VGG19', 'purple')
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for model, color in models:
        try:
            # Load per-seed results and aggregate
            results = load_exp3_results_from_per_seed(model, results_dir='results/exp3')
            aggregated = aggregate_results(results)

            # Get n_layers (or depths for backward compatibility)
            layer_key = 'n_layers' if 'n_layers' in results else 'depths'
            layers = results[layer_key]
            gaps_mean = aggregated['generalization_gaps_mean']
            gaps_std = aggregated['generalization_gaps_std']
            epochs_mean = aggregated.get('epochs_to_100pct_mean', [None] * len(layers))
            epochs_std = aggregated.get('epochs_to_100pct_std', [None] * len(layers))

            # Collect valid data - only include points where epochs_to_100pct is available
            valid_layers = []
            valid_gaps_mean = []
            valid_gaps_std = []
            valid_epochs_mean = []
            valid_epochs_std = []

            for n, gap_mean, gap_std, epoch_mean, epoch_std in zip(layers, gaps_mean, gaps_std, epochs_mean, epochs_std):
                # Only include data where epochs_to_100pct is available
                if gap_mean is not None and epoch_mean is not None:
                    valid_layers.append(n)
                    valid_gaps_mean.append(gap_mean)
                    valid_gaps_std.append(gap_std)
                    valid_epochs_mean.append(epoch_mean)
                    valid_epochs_std.append(epoch_std)

            # Plot generalization gap
            if valid_layers:
                ax1.errorbar(valid_layers, valid_gaps_mean, yerr=valid_gaps_std,
                             label=model, color=color, marker='o', capsize=5, linewidth=2)

            # Plot epochs to 100%
            if valid_layers:
                ax2.errorbar(valid_layers, valid_epochs_mean, yerr=valid_epochs_std,
                             label=model, color=color, marker='o', capsize=5, linewidth=2)

        except FileNotFoundError:
            print(f"Results not found for {model} (exp3)")

    # Configure first subplot (generalization gap)
    ax1.set_xlabel('Number of Convolutional Layers', fontsize=12)
    ax1.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax1.set_title('Generalization Gap vs Network Depth', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Configure second subplot (epochs to 100%)
    ax2.set_xlabel('Number of Convolutional Layers', fontsize=12)
    ax2.set_ylabel('Epochs to 100% Train Accuracy', fontsize=12)
    ax2.set_title('Training Speed vs Network Depth', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: VGG Variable Depth (MI=2.5 bits)', fontsize=16, y=1.00)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_3_vgg_gap_and_epochs_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 3 plot saved to plots/experiment_3_vgg_gap_and_epochs_vs_depth.png")


def plot_experiment_4():
    """Plot Experiment 4: MLP Variable - Generalization Gap and Epochs to 99% vs Number of Layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    try:
        # Load per-seed results and aggregate
        results = load_exp4_results_from_per_seed(results_dir='results/exp4')
        aggregated = aggregate_results(results)

        layers = results['n_layers']
        gaps_mean = aggregated['generalization_gaps_mean']
        gaps_std = aggregated['generalization_gaps_std']
        epochs_mean = aggregated.get('epochs_to_99pct_mean', [None] * len(layers))
        epochs_std = aggregated.get('epochs_to_99pct_std', [None] * len(layers))

        # Collect valid data - only include points where epochs_to_99pct is available
        valid_layers = []
        valid_gaps_mean = []
        valid_gaps_std = []
        valid_epochs_mean = []
        valid_epochs_std = []

        for n, gap_mean, gap_std, epoch_mean, epoch_std in zip(layers, gaps_mean, gaps_std, epochs_mean, epochs_std):
            # Only include data where epochs_to_99pct is available and n >= 2
            if gap_mean is not None and epoch_mean is not None and n >= 2:
                valid_layers.append(n)
                valid_gaps_mean.append(gap_mean)
                valid_gaps_std.append(gap_std)
                valid_epochs_mean.append(epoch_mean)
                valid_epochs_std.append(epoch_std)

        # Plot generalization gap
        if valid_layers:
            ax1.errorbar(valid_layers, valid_gaps_mean, yerr=valid_gaps_std,
                         label='MLP (256 neurons/layer)', color='blue', marker='o', capsize=5, linewidth=2)

        # Plot epochs to 99%
        if valid_layers:
            ax2.errorbar(valid_layers, valid_epochs_mean, yerr=valid_epochs_std,
                         label='MLP (256 neurons/layer)', color='blue', marker='o', capsize=5, linewidth=2)

    except FileNotFoundError:
        print(f"Results not found for MLP (exp4)")

    # Configure first subplot (generalization gap)
    ax1.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax1.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax1.set_title('Generalization Gap vs Network Depth', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Configure second subplot (epochs to 99%)
    ax2.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax2.set_ylabel('Epochs to 99% Train Accuracy', fontsize=12)
    ax2.set_title('Training Speed vs Network Depth', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 4: MLP Variable Depth (MI=2.5 bits)', fontsize=16, y=1.00)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_4_mlp_gap_and_epochs_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 4 plot saved to plots/experiment_4_mlp_gap_and_epochs_vs_depth.png")


def plot_experiment_5():
    """Plot Experiment 5: All-Conv Variable - Generalization Gap vs Number of Layers."""
    plt.figure(figsize=(12, 8))

    try:
        # Load per-seed results and aggregate
        results = load_exp5_results_from_per_seed(results_dir='results/exp5')
        aggregated = aggregate_results(results)

        layers = results['n_layers']
        gaps_mean = aggregated['generalization_gaps_mean']
        gaps_std = aggregated['generalization_gaps_std']

        valid_layers = []
        valid_gaps_mean = []
        valid_gaps_std = []

        for n, gap_mean, gap_std in zip(layers, gaps_mean, gaps_std):
            if gap_mean is not None and n >= 2:
                valid_layers.append(n)
                valid_gaps_mean.append(gap_mean)
                valid_gaps_std.append(gap_std)

        if valid_layers:
            plt.errorbar(valid_layers, valid_gaps_mean, yerr=valid_gaps_std,
                         label='All-Conv (128 channels/layer)', color='green', marker='o', capsize=5, linewidth=2)

    except FileNotFoundError:
        print(f"Results not found for All-Conv (exp5)")

    plt.xlabel('Number of Convolutional Layers', fontsize=12)
    plt.ylabel('Generalization Gap (%)', fontsize=12)
    plt.title('Experiment 5: All-Conv Generalization Gap vs Network Depth (MI=2.5 bits)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_allconv_gap_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 5 plot saved to plots/experiment_5_allconv_gap_vs_depth.png")


def main():
    print("Generating plots...")
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    plot_experiment_4()
    plot_experiment_5()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()