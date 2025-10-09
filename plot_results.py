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

    # Plot WITH special pixel (solid lines, circles)
    for model, color in models:
        try:
            results = load_exp3_results_from_per_seed(model, results_dir='results/exp3', include_nopixel=False)
            aggregated = aggregate_results(results)

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
                if gap_mean is not None and epoch_mean is not None:
                    valid_layers.append(n)
                    valid_gaps_mean.append(gap_mean)
                    valid_gaps_std.append(gap_std)
                    valid_epochs_mean.append(epoch_mean)
                    valid_epochs_std.append(epoch_std)

            # Plot generalization gap (solid line)
            if valid_layers:
                ax1.errorbar(valid_layers, valid_gaps_mean, yerr=valid_gaps_std,
                             label=f'{model} (With Pixel)', color=color, marker='o',
                             linestyle='-', capsize=5, linewidth=2)

            # Plot epochs to 100% (solid line)
            if valid_layers:
                ax2.errorbar(valid_layers, valid_epochs_mean, yerr=valid_epochs_std,
                             label=f'{model} (With Pixel)', color=color, marker='o',
                             linestyle='-', capsize=5, linewidth=2)

        except FileNotFoundError:
            print(f"Results not found for {model} with pixel (exp3)")

    # Plot WITHOUT special pixel (dashed lines, squares)
    for model, color in models:
        try:
            results_nopixel = load_exp3_results_from_per_seed(model, results_dir='results/exp3', include_nopixel=True)
            aggregated_nopixel = aggregate_results(results_nopixel)

            layer_key = 'n_layers' if 'n_layers' in results_nopixel else 'depths'
            layers_nopixel = results_nopixel[layer_key]
            gaps_mean_nopixel = aggregated_nopixel['generalization_gaps_mean']
            gaps_std_nopixel = aggregated_nopixel['generalization_gaps_std']
            epochs_mean_nopixel = aggregated_nopixel.get('epochs_to_100pct_mean', [None] * len(layers_nopixel))
            epochs_std_nopixel = aggregated_nopixel.get('epochs_to_100pct_std', [None] * len(layers_nopixel))

            # Collect valid data
            valid_layers_nopixel = []
            valid_gaps_mean_nopixel = []
            valid_gaps_std_nopixel = []
            valid_epochs_mean_nopixel = []
            valid_epochs_std_nopixel = []

            for n, gap_mean, gap_std, epoch_mean, epoch_std in zip(layers_nopixel, gaps_mean_nopixel, gaps_std_nopixel, epochs_mean_nopixel, epochs_std_nopixel):
                if gap_mean is not None and epoch_mean is not None:
                    valid_layers_nopixel.append(n)
                    valid_gaps_mean_nopixel.append(gap_mean)
                    valid_gaps_std_nopixel.append(gap_std)
                    valid_epochs_mean_nopixel.append(epoch_mean)
                    valid_epochs_std_nopixel.append(epoch_std)

            # Plot generalization gap (dashed line)
            if valid_layers_nopixel:
                ax1.errorbar(valid_layers_nopixel, valid_gaps_mean_nopixel, yerr=valid_gaps_std_nopixel,
                             label=f'{model} (No Pixel)', color=color, marker='s',
                             linestyle='--', capsize=5, linewidth=2)

            # Plot epochs to 100% (dashed line)
            if valid_layers_nopixel:
                ax2.errorbar(valid_layers_nopixel, valid_epochs_mean_nopixel, yerr=valid_epochs_std_nopixel,
                             label=f'{model} (No Pixel)', color=color, marker='s',
                             linestyle='--', capsize=5, linewidth=2)

        except FileNotFoundError:
            print(f"Results not found for {model} without pixel (exp3)")

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
    """Plot Experiment 4: MLP Variable - Generalization Gap (epoch 5 and final) and Epochs to 99% vs Number of Layers."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Load and plot WITH special pixel (blue)
    try:
        results = load_exp4_results_from_per_seed(results_dir='results/exp4', include_nopixel=False)
        aggregated = aggregate_results(results)

        layers = results['n_layers']
        gap_epoch5_mean = aggregated.get('gap_epoch5_mean', [None] * len(layers))
        gap_epoch5_std = aggregated.get('gap_epoch5_std', [None] * len(layers))
        gaps_mean = aggregated['generalization_gaps_mean']
        gaps_std = aggregated['generalization_gaps_std']
        epochs_mean = aggregated.get('epochs_to_99pct_mean', [None] * len(layers))
        epochs_std = aggregated.get('epochs_to_99pct_std', [None] * len(layers))

        # Collect valid data for gap at epoch 5
        valid_layers_gap5 = []
        valid_gap5_mean = []
        valid_gap5_std = []

        for n, gap_mean, gap_std in zip(layers, gap_epoch5_mean, gap_epoch5_std):
            if gap_mean is not None and n >= 2:
                valid_layers_gap5.append(n)
                valid_gap5_mean.append(gap_mean)
                valid_gap5_std.append(gap_std)

        # Collect valid data for generalization gap (final)
        valid_layers_gap = []
        valid_gaps_mean = []
        valid_gaps_std = []

        for n, gap_mean, gap_std in zip(layers, gaps_mean, gaps_std):
            if gap_mean is not None and n >= 2:
                valid_layers_gap.append(n)
                valid_gaps_mean.append(gap_mean)
                valid_gaps_std.append(gap_std)

        # Collect valid data for epochs to 99%
        valid_layers_epochs = []
        valid_epochs_mean = []
        valid_epochs_std = []

        for n, epoch_mean, epoch_std in zip(layers, epochs_mean, epochs_std):
            if epoch_mean is not None and n >= 2:
                valid_layers_epochs.append(n)
                valid_epochs_mean.append(epoch_mean)
                valid_epochs_std.append(epoch_std)

        # Plot gap at epoch 5 (blue)
        if valid_layers_gap5:
            ax1.errorbar(valid_layers_gap5, valid_gap5_mean, yerr=valid_gap5_std,
                         label='With Special Pixel', color='blue', marker='o', capsize=5, linewidth=2)

        # Plot final generalization gap (blue)
        if valid_layers_gap:
            ax2.errorbar(valid_layers_gap, valid_gaps_mean, yerr=valid_gaps_std,
                         label='With Special Pixel', color='blue', marker='o', capsize=5, linewidth=2)

        # Plot epochs to 99% (blue)
        if valid_layers_epochs:
            ax3.errorbar(valid_layers_epochs, valid_epochs_mean, yerr=valid_epochs_std,
                         label='With Special Pixel', color='blue', marker='o', capsize=5, linewidth=2)

    except FileNotFoundError:
        print(f"Results not found for MLP with pixel (exp4)")

    # Load and plot WITHOUT special pixel (red)
    try:
        results_nopixel = load_exp4_results_from_per_seed(results_dir='results/exp4', include_nopixel=True)
        aggregated_nopixel = aggregate_results(results_nopixel)

        layers_nopixel = results_nopixel['n_layers']
        gap_epoch5_mean_nopixel = aggregated_nopixel.get('gap_epoch5_mean', [None] * len(layers_nopixel))
        gap_epoch5_std_nopixel = aggregated_nopixel.get('gap_epoch5_std', [None] * len(layers_nopixel))
        gaps_mean_nopixel = aggregated_nopixel['generalization_gaps_mean']
        gaps_std_nopixel = aggregated_nopixel['generalization_gaps_std']
        epochs_mean_nopixel = aggregated_nopixel.get('epochs_to_99pct_mean', [None] * len(layers_nopixel))
        epochs_std_nopixel = aggregated_nopixel.get('epochs_to_99pct_std', [None] * len(layers_nopixel))

        # Collect valid data for gap at epoch 5
        valid_layers_gap5_nopixel = []
        valid_gap5_mean_nopixel = []
        valid_gap5_std_nopixel = []

        for n, gap_mean, gap_std in zip(layers_nopixel, gap_epoch5_mean_nopixel, gap_epoch5_std_nopixel):
            if gap_mean is not None and n >= 2:
                valid_layers_gap5_nopixel.append(n)
                valid_gap5_mean_nopixel.append(gap_mean)
                valid_gap5_std_nopixel.append(gap_std)

        # Collect valid data for generalization gap (final)
        valid_layers_gap_nopixel = []
        valid_gaps_mean_nopixel = []
        valid_gaps_std_nopixel = []

        for n, gap_mean, gap_std in zip(layers_nopixel, gaps_mean_nopixel, gaps_std_nopixel):
            if gap_mean is not None and n >= 2:
                valid_layers_gap_nopixel.append(n)
                valid_gaps_mean_nopixel.append(gap_mean)
                valid_gaps_std_nopixel.append(gap_std)

        # Collect valid data for epochs to 99%
        valid_layers_epochs_nopixel = []
        valid_epochs_mean_nopixel = []
        valid_epochs_std_nopixel = []

        for n, epoch_mean, epoch_std in zip(layers_nopixel, epochs_mean_nopixel, epochs_std_nopixel):
            if epoch_mean is not None and n >= 2:
                valid_layers_epochs_nopixel.append(n)
                valid_epochs_mean_nopixel.append(epoch_mean)
                valid_epochs_std_nopixel.append(epoch_std)

        # Plot gap at epoch 5 (red)
        if valid_layers_gap5_nopixel:
            ax1.errorbar(valid_layers_gap5_nopixel, valid_gap5_mean_nopixel, yerr=valid_gap5_std_nopixel,
                         label='Without Special Pixel', color='red', marker='s', capsize=5, linewidth=2)

        # Plot final generalization gap (red)
        if valid_layers_gap_nopixel:
            ax2.errorbar(valid_layers_gap_nopixel, valid_gaps_mean_nopixel, yerr=valid_gaps_std_nopixel,
                         label='Without Special Pixel', color='red', marker='s', capsize=5, linewidth=2)

        # Plot epochs to 99% (red)
        if valid_layers_epochs_nopixel:
            ax3.errorbar(valid_layers_epochs_nopixel, valid_epochs_mean_nopixel, yerr=valid_epochs_std_nopixel,
                         label='Without Special Pixel', color='red', marker='s', capsize=5, linewidth=2)

    except FileNotFoundError:
        print(f"Results not found for MLP without pixel (exp4)")

    # Configure first subplot (gap at epoch 5)
    ax1.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax1.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax1.set_title('Generalization Gap at Epoch 5', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Configure second subplot (final generalization gap)
    ax2.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax2.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax2.set_title('Final Generalization Gap', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Configure third subplot (epochs to 99%)
    ax3.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax3.set_ylabel('Epochs to 99% Train Accuracy', fontsize=12)
    ax3.set_title('Training Speed vs Network Depth', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

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


def plot_experiment_4_occlusion():
    """
    Plot Experiment 4 occlusion sensitivity visualization.
    2 rows × 6 columns showing occlusion maps at epoch 5 (top) and final epoch (bottom)
    for depths: 1, 5, 10, 15, 20, 25 layers.
    """
    import matplotlib.gridspec as gridspec

    target_depths = [1, 5, 10, 15, 20, 25]
    target_class = 5  # Visualize digit 5

    # Create figure with 2 rows (epoch5, final) and 6 columns (depths)
    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.3)

    results_dir = 'results/exp4'

    # Try to load results for each target depth
    for col_idx, depth in enumerate(target_depths):
        # Try to find a result file for this depth (any seed)
        file_pattern = f"mlp_layers{depth}_seed*_results.npz"
        import glob
        matching_files = glob.glob(os.path.join(results_dir, f"mlp_layers{depth}_seed*_results.npz"))

        if not matching_files:
            print(f"No results found for depth {depth}")
            continue

        # Load first matching file
        result_file = matching_files[0]
        data = np.load(result_file, allow_pickle=True)

        # Extract occlusion data
        if 'occlusion_maps_epoch5' not in data or 'occlusion_maps_final' not in data:
            print(f"No occlusion data found for depth {depth}")
            continue

        occlusion_epoch5 = data['occlusion_maps_epoch5']
        occlusion_final = data['occlusion_maps_final']
        sample_image = data['sample_images_epoch5'][target_class]  # Same image for both epochs

        # Get occlusion maps for target class
        occ_map_epoch5 = occlusion_epoch5[target_class]
        occ_map_final = occlusion_final[target_class]

        # Normalize occlusion maps to [0, 1] for visualization
        # Higher values = more important
        occ_epoch5_norm = (occ_map_epoch5 - occ_map_epoch5.min()) / (occ_map_epoch5.max() - occ_map_epoch5.min() + 1e-10)
        occ_final_norm = (occ_map_final - occ_map_final.min()) / (occ_map_final.max() - occ_map_final.min() + 1e-10)

        # Plot epoch 5 (top row)
        ax_epoch5 = fig.add_subplot(gs[0, col_idx])
        ax_epoch5.imshow(sample_image, cmap='gray', alpha=0.7)
        im_epoch5 = ax_epoch5.imshow(occ_epoch5_norm, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        ax_epoch5.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nEpoch 5', fontsize=10)
        ax_epoch5.axis('off')

        # Plot final epoch (bottom row)
        ax_final = fig.add_subplot(gs[1, col_idx])
        ax_final.imshow(sample_image, cmap='gray', alpha=0.7)
        im_final = ax_final.imshow(occ_final_norm, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        ax_final.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nFinal Epoch', fontsize=10)
        ax_final.axis('off')

    # Add colorbar
    fig.colorbar(im_final, ax=fig.get_axes(), orientation='vertical',
                 label='Occlusion Sensitivity (normalized)', shrink=0.6, pad=0.02)

    fig.suptitle(f'Experiment 4: Occlusion Sensitivity for Digit {target_class}',
                 fontsize=16, y=0.98)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_4_occlusion_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 4 occlusion plot saved to plots/experiment_4_occlusion_sensitivity.png")


def main():
    print("Generating plots...")
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    plot_experiment_4()
    plot_experiment_4_occlusion()
    plot_experiment_5()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()