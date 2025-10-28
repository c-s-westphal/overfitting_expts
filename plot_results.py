#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.results_utils import load_experiment_results, aggregate_results, load_exp1_results_from_per_seed, load_exp2_results_from_per_seed, load_exp3_results_from_per_seed, load_exp4_results_from_per_seed, load_exp5_results_from_per_seed

# Set up Times New Roman font for publication-quality plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def plot_experiment_1():
    """Plot Experiment 1: Generalization gap vs training set size with publication-quality styling."""
    models = ['VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19']

    # Publication-quality color scheme
    colors = {
        'VGG9': '#F39B7F',   # Peach
        'VGG11': '#E64B35',  # Red-orange
        'VGG13': '#4DBBD5',  # Cyan
        'VGG16': '#00A087',  # Teal
        'VGG19': '#3C5488'   # Blue
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in models:
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
                valid_sizes = np.array(valid_sizes)
                valid_gaps_mean = np.array(valid_gaps_mean)
                valid_gaps_std = np.array(valid_gaps_std)

                # Plot line without markers
                ax.plot(valid_sizes, valid_gaps_mean,
                       color=colors[model],
                       linewidth=2.5,
                       label=model,
                       alpha=0.85)

                # Add shaded error region
                ax.fill_between(valid_sizes,
                               valid_gaps_mean - valid_gaps_std,
                               valid_gaps_mean + valid_gaps_std,
                               color=colors[model],
                               alpha=0.2)

        except FileNotFoundError:
            print(f"Results not found for {model}")

    # Publication-quality styling
    ax.set_xlabel('Training Set Size', fontsize=16, fontweight='normal')
    ax.set_ylabel('Generalization Gap (%)', fontsize=16, fontweight='normal')
    ax.set_title('Generalization Gap vs Training Set Size', fontsize=16, fontweight='bold', pad=10)
    ax.set_xscale('log')

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend with updated styling
    ax.legend(fontsize=12, loc='best', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)

    # Remove grid lines
    ax.grid(False)

    # Black border (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Set background color to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_1_generalization_gap.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Experiment 1 plot saved to plots/experiment_1_generalization_gap.png")


def plot_experiment_2():
    """Plot Experiment 2: Generalization gap vs MI of special pixel with publication-quality styling."""
    models = ['VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19']

    # Publication-quality color scheme
    colors = {
        'VGG9': '#F39B7F',   # Peach
        'VGG11': '#E64B35',  # Red-orange
        'VGG13': '#4DBBD5',  # Cyan
        'VGG16': '#00A087',  # Teal
        'VGG19': '#3C5488'   # Blue
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in models:
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
                valid_mi = np.array(valid_mi)
                valid_gaps_mean = np.array(valid_gaps_mean)
                valid_gaps_std = np.array(valid_gaps_std)

                # Plot line without markers
                ax.plot(valid_mi, valid_gaps_mean,
                       color=colors[model],
                       linewidth=2.5,
                       label=model,
                       alpha=0.85)

                # Add shaded error region
                ax.fill_between(valid_mi,
                               valid_gaps_mean - valid_gaps_std,
                               valid_gaps_mean + valid_gaps_std,
                               color=colors[model],
                               alpha=0.2)

        except FileNotFoundError:
            print(f"Results not found for {model}")

    # Publication-quality styling
    ax.set_xlabel('Mutual Information (bits)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Generalization Gap (%)', fontsize=16, fontweight='normal')
    ax.set_title('Generalization Gap vs MI of Subsets (LB)', fontsize=16, fontweight='bold', pad=10)

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend with updated styling
    ax.legend(fontsize=12, loc='best', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)

    # Remove grid lines
    ax.grid(False)

    # Black border (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Set background color to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_2_mi_vs_gap.png', dpi=600, bbox_inches='tight', facecolor='white')
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


def plot_experiment_3_occlusion():
    """Plot Experiment 3 occlusion sensitivity (with pixel)."""
    import matplotlib.gridspec as gridspec
    import glob

    results_dir = 'results/exp3'

    # We'll plot for specific VGG architectures and layer counts
    # Format: (arch_name, [evenly spaced layers])
    # Using up to 5 evenly spaced layers (4 for VGG11 due to smaller range)
    configs = [
        ('vgg11var', [4, 5, 7, 8]),        # VGG11: 4-8 layers (4 points)
        ('vgg13var', [4, 6, 7, 9, 10]),    # VGG13: 4-10 layers (5 points)
        ('vgg16var', [4, 6, 9, 11, 13]),   # VGG16: 4-13 layers (5 points)
        ('vgg19var', [4, 7, 10, 13, 16]),  # VGG19: 4-16 layers (5 points)
    ]

    for arch_name, target_layers in configs:
        fig = plt.figure(figsize=(5 * len(target_layers), 8))
        gs = gridspec.GridSpec(2, len(target_layers), figure=fig, hspace=0.3, wspace=0.3)

        found_any = False
        im_final = None

        for col_idx, n_layers in enumerate(target_layers):
            # Load all seeds and average occlusion maps
            all_files = glob.glob(os.path.join(results_dir, f"{arch_name}_layers{n_layers}_seed*_results.npz"))
            matching_files = [f for f in all_files if 'nopixel' not in f]

            if not matching_files:
                print(f"No results found for {arch_name} layers {n_layers}")
                continue

            # Collect occlusion maps across all seeds AND all classes
            epoch1_maps_all_classes = []
            final_maps_all_classes = []
            sample_image = None

            for result_file in matching_files:
                data = np.load(result_file, allow_pickle=True)

                # Check if occlusion data exists
                if 'occlusion_maps_epoch1' not in data or 'occlusion_maps_final' not in data:
                    continue

                occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 32, 32)
                occlusion_final = data['occlusion_maps_final']

                # Average across all 10 classes for this seed
                epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))
                final_maps_all_classes.append(np.mean(occlusion_final, axis=0))

                # Use seed 0's first sample image (class 0) just for display
                if sample_image is None and 'seed0' in result_file:
                    sample_image = data['sample_images_epoch1'][0]

            if not epoch1_maps_all_classes:
                print(f"No occlusion data found for {arch_name} layers {n_layers}")
                continue

            # Use first available sample image if seed0 not found
            if sample_image is None:
                data = np.load(matching_files[0], allow_pickle=True)
                if 'sample_images_epoch1' in data:
                    sample_image = data['sample_images_epoch1'][0]

            # Average occlusion maps across all seeds (already averaged across classes)
            occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)
            occ_map_final = np.mean(final_maps_all_classes, axis=0)

            # Apply power transform to compress dynamic range
            power = 0.5  # Square root transform

            # Shift to non-negative before power transform
            occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
            occ_epoch1_transformed = occ_epoch1_shifted ** power
            occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

            occ_final_shifted = occ_map_final - occ_map_final.min()
            occ_final_transformed = occ_final_shifted ** power
            occ_final_norm = (occ_final_transformed - occ_final_transformed.min()) / (occ_final_transformed.max() - occ_final_transformed.min() + 1e-10)

            if sample_image is not None:
                # Convert RGB image from (3, 32, 32) to (32, 32, 3) for display
                sample_image_display = np.transpose(sample_image, (1, 2, 0))
                # Normalize to [0, 1] for display
                sample_image_display = (sample_image_display - sample_image_display.min()) / (sample_image_display.max() - sample_image_display.min() + 1e-10)

            # Plot epoch 1 (top row)
            ax_epoch1 = fig.add_subplot(gs[0, col_idx])
            if sample_image is not None:
                ax_epoch1.imshow(sample_image_display)
            im_epoch1 = ax_epoch1.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
            ax_epoch1.set_title(f'{n_layers} layers\nEpoch 1', fontsize=10)
            ax_epoch1.axis('off')

            # Plot final epoch (bottom row)
            ax_final = fig.add_subplot(gs[1, col_idx])
            if sample_image is not None:
                ax_final.imshow(sample_image_display)
            im_final = ax_final.imshow(occ_final_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
            ax_final.set_title(f'{n_layers} layers\nFinal Epoch', fontsize=10)
            ax_final.axis('off')

            found_any = True

        # Check if we found any data
        if not found_any:
            plt.close(fig)
            print(f"No occlusion data found for {arch_name}")
            continue

        # Add colorbar
        if im_final is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im_final, cax=cbar_ax)
            cbar.set_label('Occlusion Sensitivity (normalized)', rotation=270, labelpad=20)

        arch_display = arch_name.replace('var', '').upper()
        fig.suptitle(f'Experiment 3: {arch_display} Occlusion Sensitivity (Averaged Across All Classes)',
                     fontsize=14, y=0.98)

        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/experiment_3_{arch_name}_occlusion_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Experiment 3 {arch_name} occlusion plot saved to plots/experiment_3_{arch_name}_occlusion_sensitivity.png")


def plot_experiment_3_occlusion_nopixel():
    """Plot Experiment 3 occlusion sensitivity (without pixel)."""
    import matplotlib.gridspec as gridspec
    import glob

    results_dir = 'results/exp3'

    # We'll plot for specific VGG architectures and layer counts
    # Format: (arch_name, [evenly spaced layers])
    # Using up to 5 evenly spaced layers (4 for VGG11 due to smaller range)
    configs = [
        ('vgg11var', [4, 5, 7, 8]),        # VGG11: 4-8 layers (4 points)
        ('vgg13var', [4, 6, 7, 9, 10]),    # VGG13: 4-10 layers (5 points)
        ('vgg16var', [4, 6, 9, 11, 13]),   # VGG16: 4-13 layers (5 points)
        ('vgg19var', [4, 7, 10, 13, 16]),  # VGG19: 4-16 layers (5 points)
    ]

    for arch_name, target_layers in configs:
        fig = plt.figure(figsize=(5 * len(target_layers), 8))
        gs = gridspec.GridSpec(2, len(target_layers), figure=fig, hspace=0.3, wspace=0.3)

        found_any = False
        im_final = None

        for col_idx, n_layers in enumerate(target_layers):
            # Load all seeds and average occlusion maps (NOPIXEL FILES)
            all_files = glob.glob(os.path.join(results_dir, f"{arch_name}_layers{n_layers}_seed*_nopixel_results.npz"))
            matching_files = all_files

            if not matching_files:
                print(f"No nopixel results found for {arch_name} layers {n_layers}")
                continue

            # Collect occlusion maps across all seeds AND all classes
            epoch1_maps_all_classes = []
            final_maps_all_classes = []
            sample_image = None

            for result_file in matching_files:
                data = np.load(result_file, allow_pickle=True)

                # Check if occlusion data exists
                if 'occlusion_maps_epoch1' not in data or 'occlusion_maps_final' not in data:
                    continue

                occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 32, 32)
                occlusion_final = data['occlusion_maps_final']

                # Average across all 10 classes for this seed
                epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))
                final_maps_all_classes.append(np.mean(occlusion_final, axis=0))

                # Use seed 0's first sample image (class 0) just for display
                if sample_image is None and 'seed0' in result_file:
                    sample_image = data['sample_images_epoch1'][0]

            if not epoch1_maps_all_classes:
                print(f"No occlusion data found for {arch_name} layers {n_layers} (nopixel)")
                continue

            # Use first available sample image if seed0 not found
            if sample_image is None:
                data = np.load(matching_files[0], allow_pickle=True)
                if 'sample_images_epoch1' in data:
                    sample_image = data['sample_images_epoch1'][0]

            # Average occlusion maps across all seeds (already averaged across classes)
            occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)
            occ_map_final = np.mean(final_maps_all_classes, axis=0)

            # Apply power transform to compress dynamic range
            power = 0.5  # Square root transform

            # Shift to non-negative before power transform
            occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
            occ_epoch1_transformed = occ_epoch1_shifted ** power
            occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

            occ_final_shifted = occ_map_final - occ_map_final.min()
            occ_final_transformed = occ_final_shifted ** power
            occ_final_norm = (occ_final_transformed - occ_final_transformed.min()) / (occ_final_transformed.max() - occ_final_transformed.min() + 1e-10)

            if sample_image is not None:
                # Convert RGB image from (3, 32, 32) to (32, 32, 3) for display
                sample_image_display = np.transpose(sample_image, (1, 2, 0))
                # Normalize to [0, 1] for display
                sample_image_display = (sample_image_display - sample_image_display.min()) / (sample_image_display.max() - sample_image_display.min() + 1e-10)

            # Plot epoch 1 (top row)
            ax_epoch1 = fig.add_subplot(gs[0, col_idx])
            if sample_image is not None:
                ax_epoch1.imshow(sample_image_display)
            im_epoch1 = ax_epoch1.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
            ax_epoch1.set_title(f'{n_layers} layers\nEpoch 1', fontsize=10)
            ax_epoch1.axis('off')

            # Plot final epoch (bottom row)
            ax_final = fig.add_subplot(gs[1, col_idx])
            if sample_image is not None:
                ax_final.imshow(sample_image_display)
            im_final = ax_final.imshow(occ_final_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
            ax_final.set_title(f'{n_layers} layers\nFinal Epoch', fontsize=10)
            ax_final.axis('off')

            found_any = True

        # Check if we found any data
        if not found_any:
            plt.close(fig)
            print(f"No occlusion data found for {arch_name} (nopixel)")
            continue

        # Add colorbar
        if im_final is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im_final, cax=cbar_ax)
            cbar.set_label('Occlusion Sensitivity (normalized)', rotation=270, labelpad=20)

        arch_display = arch_name.replace('var', '').upper()
        fig.suptitle(f'Experiment 3: {arch_display} Occlusion Sensitivity - No Special Pixel (Averaged Across All Classes)',
                     fontsize=14, y=0.98)

        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/experiment_3_{arch_name}_occlusion_sensitivity_nopixel.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Experiment 3 {arch_name} nopixel occlusion plot saved to plots/experiment_3_{arch_name}_occlusion_sensitivity_nopixel.png")


def plot_experiment_4():
    """Plot Experiment 4: MLP Variable - Generalization Gap (epoch 5 and final) and Epochs to 99% vs Number of Layers."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Load and plot WITH special pixel (blue)
    try:
        results = load_exp4_results_from_per_seed(results_dir='results/exp4', include_nopixel=False)
        aggregated = aggregate_results(results)

        layers = results['n_layers']
        gap_epoch1_mean = aggregated.get('gap_epoch1_mean', [None] * len(layers))
        gap_epoch1_std = aggregated.get('gap_epoch1_std', [None] * len(layers))
        gaps_mean = aggregated['generalization_gaps_mean']
        gaps_std = aggregated['generalization_gaps_std']
        epochs_mean = aggregated.get('epochs_to_99pct_mean', [None] * len(layers))
        epochs_std = aggregated.get('epochs_to_99pct_std', [None] * len(layers))

        # Collect valid data for gap at epoch 1
        valid_layers_gap2 = []
        valid_gap2_mean = []
        valid_gap2_std = []

        for n, gap_mean, gap_std in zip(layers, gap_epoch1_mean, gap_epoch1_std):
            if gap_mean is not None and n >= 2:
                valid_layers_gap2.append(n)
                valid_gap2_mean.append(gap_mean)
                valid_gap2_std.append(gap_std)

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

        # Plot gap at epoch 1 (blue)
        if valid_layers_gap2:
            ax1.errorbar(valid_layers_gap2, valid_gap2_mean, yerr=valid_gap2_std,
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
        gap_epoch1_mean_nopixel = aggregated_nopixel.get('gap_epoch1_mean', [None] * len(layers_nopixel))
        gap_epoch1_std_nopixel = aggregated_nopixel.get('gap_epoch1_std', [None] * len(layers_nopixel))
        gaps_mean_nopixel = aggregated_nopixel['generalization_gaps_mean']
        gaps_std_nopixel = aggregated_nopixel['generalization_gaps_std']
        epochs_mean_nopixel = aggregated_nopixel.get('epochs_to_99pct_mean', [None] * len(layers_nopixel))
        epochs_std_nopixel = aggregated_nopixel.get('epochs_to_99pct_std', [None] * len(layers_nopixel))

        # Collect valid data for gap at epoch 1
        valid_layers_gap2_nopixel = []
        valid_gap2_mean_nopixel = []
        valid_gap2_std_nopixel = []

        for n, gap_mean, gap_std in zip(layers_nopixel, gap_epoch1_mean_nopixel, gap_epoch1_std_nopixel):
            if gap_mean is not None and n >= 2:
                valid_layers_gap2_nopixel.append(n)
                valid_gap2_mean_nopixel.append(gap_mean)
                valid_gap2_std_nopixel.append(gap_std)

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

        # Plot gap at epoch 1 (red)
        if valid_layers_gap2_nopixel:
            ax1.errorbar(valid_layers_gap2_nopixel, valid_gap2_mean_nopixel, yerr=valid_gap2_std_nopixel,
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

    # Configure first subplot (gap at epoch 1)
    ax1.set_xlabel('Number of Hidden Layers', fontsize=12)
    ax1.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax1.set_title('Generalization Gap at Epoch 1', fontsize=14)
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
    #plt.savefig('plots/experiment_4_mlp_gap_and_epochs_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 4 plot saved to plots/experiment_4_mlp_gap_and_epochs_vs_depth.png")


def plot_experiment_5():
    """Plot Experiment 5: MNIST MLP - Layer 1 Subset Synergy vs Generalization Gap."""
    import glob

    # Publication-quality color scheme for layers
    colors = {
        1: '#F39B7F',   # Peach
        2: '#E64B35',   # Red-orange
        3: '#4DBBD5',   # Cyan
        4: '#00A087',   # Teal
        5: '#3C5488'    # Blue
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Load data for each layer configuration
    results_dir = 'results/exp5'
    layer_configs = [1, 2, 3, 4, 5]

    marker = 'o'
    marker_size = 120

    # Collect all data points for overall trend line
    all_gen_gaps = []
    all_mi_diffs = []

    for n_layers in layer_configs:
        # Load all seeds for this layer config
        pattern = os.path.join(results_dir, f"mlp_mnist_layers{n_layers}_seed*_results.npz")
        files = glob.glob(pattern)

        if not files:
            print(f"No results found for {n_layers} layers")
            continue

        gen_gaps = []
        mi_diffs = []

        for fpath in files:
            data = np.load(fpath, allow_pickle=True)
            gen_gaps.append(float(data['final_gen_gap']))
            mi_diffs.append(float(data['final_mi_diff']))

        # Add to global lists for trend line
        all_gen_gaps.extend(gen_gaps)
        all_mi_diffs.extend(mi_diffs)

        # Get color for this architecture
        color = colors[n_layers]

        # Plot scatter points
        ax.scatter(gen_gaps, mi_diffs,
                  c=color, marker=marker, s=marker_size, alpha=0.75,
                  edgecolors='black', linewidths=1.0,
                  label=f'{n_layers} layer{"s" if n_layers > 1 else ""}',
                  zorder=3)

    # Add overall trend line if we have data
    if len(all_gen_gaps) > 1:
        # Fit linear regression
        z = np.polyfit(all_gen_gaps, all_mi_diffs, 1)
        p = np.poly1d(z)

        # Calculate R²
        y_pred = p(all_gen_gaps)
        ss_res = np.sum((np.array(all_mi_diffs) - y_pred) ** 2)
        ss_tot = np.sum((np.array(all_mi_diffs) - np.mean(all_mi_diffs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Plot trend line
        x_trend = np.linspace(min(all_gen_gaps), max(all_gen_gaps), 100)
        y_trend = p(x_trend)
        ax.plot(x_trend, y_trend, 'k-', linewidth=2.5, alpha=0.5, zorder=2)

        # Add R² text on the graph (lower left)
        ax.text(0.05, 0.05, f'$R^2$ = {r_squared:.3f}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                       edgecolor='black', linewidth=1.2, pad=0.5))

    # Publication-quality styling
    ax.set_xlabel('Generalization Gap (%)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Layer 1 Subset Synergy', fontsize=16, fontweight='normal')
    ax.set_title('Demonstrating Corollary 1', fontsize=16, fontweight='bold', pad=8)

    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend
    ax.legend(loc='best', fontsize=12, frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)

    # Remove grid lines
    ax.grid(False)

    # Black border (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Set background color to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_5_mnist_mi_diff_vs_gen_gap_by_layers.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Experiment 5 plot saved to plots/experiment_5_mnist_mi_diff_vs_gen_gap_by_layers.png")


def plot_experiment_4_occlusion():
    """
    Plot Experiment 4 occlusion sensitivity visualization.
    2 rows × 5 columns showing occlusion maps at epoch 1 (top) and final epoch (bottom)
    for depths: 5, 10, 15, 20, 25 layers.
    """
    import matplotlib.gridspec as gridspec

    target_depths = [5, 10, 15, 20, 25]

    # Create figure with 2 rows (epoch1, final) and 5 columns (depths)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    results_dir = 'results/exp4'

    # Track if we found any data
    found_any = False
    im_final = None

    # Try to load results for each target depth
    for col_idx, depth in enumerate(target_depths):
        # Load all seeds and average occlusion maps
        import glob
        all_files = glob.glob(os.path.join(results_dir, f"mlp_layers{depth}_seed*_results.npz"))
        matching_files = [f for f in all_files if 'nopixel' not in f]

        if not matching_files:
            print(f"No results found for depth {depth}")
            continue

        # Collect occlusion maps across all seeds AND all classes
        epoch1_maps_all_classes = []
        final_maps_all_classes = []
        sample_image = None

        for result_file in matching_files:
            data = np.load(result_file, allow_pickle=True)

            # Extract occlusion data
            if 'occlusion_maps_epoch1' not in data or 'occlusion_maps_final' not in data:
                continue

            occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 28, 28)
            occlusion_final = data['occlusion_maps_final']    # Shape: (10, 28, 28)

            # Average across all 10 classes for this seed
            epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))
            final_maps_all_classes.append(np.mean(occlusion_final, axis=0))

            # Use seed 0's first sample image (class 0) just for display
            if sample_image is None and 'seed0' in result_file:
                sample_image = data['sample_images_epoch1'][0]

        if not epoch1_maps_all_classes:
            print(f"No occlusion data found for depth {depth}")
            continue

        # Use first available sample image if seed0 not found
        if sample_image is None:
            data = np.load(matching_files[0], allow_pickle=True)
            sample_image = data['sample_images_epoch1'][0]

        # Average occlusion maps across all seeds (already averaged across classes)
        occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)
        occ_map_final = np.mean(final_maps_all_classes, axis=0)

        # Apply power transform to compress dynamic range (reduces dominance of extreme values)
        # Then normalize to [0, 1] using min-max
        # Higher values = more important
        power = 0.5  # Square root transform

        # Shift to non-negative before power transform (to avoid NaN from negative values)
        occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
        occ_epoch1_transformed = occ_epoch1_shifted ** power
        occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

        occ_final_shifted = occ_map_final - occ_map_final.min()
        occ_final_transformed = occ_final_shifted ** power
        occ_final_norm = (occ_final_transformed - occ_final_transformed.min()) / (occ_final_transformed.max() - occ_final_transformed.min() + 1e-10)

        # Plot epoch 1 (top row)
        ax_epoch1 = fig.add_subplot(gs[0, col_idx])
        ax_epoch1.imshow(sample_image, cmap='gray', alpha=0.7)
        im_epoch1 = ax_epoch1.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
        ax_epoch1.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nEpoch 1', fontsize=10)
        ax_epoch1.axis('off')

        # Plot final epoch (bottom row)
        ax_final = fig.add_subplot(gs[1, col_idx])
        ax_final.imshow(sample_image, cmap='gray', alpha=0.7)
        im_final = ax_final.imshow(occ_final_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
        ax_final.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nFinal Epoch', fontsize=10)
        ax_final.axis('off')

        found_any = True

    # Check if we found any data
    if not found_any:
        plt.close(fig)
        print("No occlusion data found for any target depths. Skipping occlusion plot.")
        print("Run experiments with new code to generate occlusion data.")
        return

    # Add colorbar
    fig.colorbar(im_final, ax=fig.get_axes(), orientation='vertical',
                 label='Occlusion Sensitivity (normalized)', shrink=0.6, pad=0.02)

    fig.suptitle(f'Experiment 4: Occlusion Sensitivity (Averaged Across All Digits)',
                 fontsize=16, y=0.98)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_4_occlusion_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 4 occlusion plot saved to plots/experiment_4_occlusion_sensitivity.png")


def plot_experiment_4_occlusion_nopixel():
    """
    Plot Experiment 4 occlusion sensitivity visualization for NO PIXEL experiments.
    2 rows × 5 columns showing occlusion maps at epoch 1 (top) and final epoch (bottom)
    for depths: 5, 10, 15, 20, 25 layers.
    """
    import matplotlib.gridspec as gridspec

    target_depths = [5, 10, 15, 20, 25]

    # Create figure with 2 rows (epoch1, final) and 5 columns (depths)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    results_dir = 'results/exp4'

    # Track if we found any data
    found_any = False
    im_final = None

    # Try to load results for each target depth
    for col_idx, depth in enumerate(target_depths):
        # Load all seeds and average occlusion maps
        import glob
        matching_files = glob.glob(os.path.join(results_dir, f"mlp_layers{depth}_seed*_nopixel_results.npz"))

        if not matching_files:
            print(f"No nopixel results found for depth {depth}")
            continue

        # Collect occlusion maps across all seeds AND all classes
        epoch1_maps_all_classes = []
        final_maps_all_classes = []
        sample_image = None

        for result_file in matching_files:
            data = np.load(result_file, allow_pickle=True)

            # Extract occlusion data
            if 'occlusion_maps_epoch1' not in data or 'occlusion_maps_final' not in data:
                continue

            occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 28, 28)
            occlusion_final = data['occlusion_maps_final']    # Shape: (10, 28, 28)

            # Average across all 10 classes for this seed
            epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))
            final_maps_all_classes.append(np.mean(occlusion_final, axis=0))

            # Use seed 0's first sample image (class 0) just for display
            if sample_image is None and 'seed0' in result_file:
                sample_image = data['sample_images_epoch1'][0]

        if not epoch1_maps_all_classes:
            print(f"No nopixel occlusion data found for depth {depth}")
            continue

        # Use first available sample image if seed0 not found
        if sample_image is None:
            data = np.load(matching_files[0], allow_pickle=True)
            sample_image = data['sample_images_epoch1'][0]

        # Average occlusion maps across all seeds (already averaged across classes)
        occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)
        occ_map_final = np.mean(final_maps_all_classes, axis=0)

        # Apply power transform to compress dynamic range (reduces dominance of extreme values)
        # Then normalize to [0, 1] using min-max
        # Higher values = more important
        power = 0.5  # Square root transform

        # Shift to non-negative before power transform (to avoid NaN from negative values)
        occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
        occ_epoch1_transformed = occ_epoch1_shifted ** power
        occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

        occ_final_shifted = occ_map_final - occ_map_final.min()
        occ_final_transformed = occ_final_shifted ** power
        occ_final_norm = (occ_final_transformed - occ_final_transformed.min()) / (occ_final_transformed.max() - occ_final_transformed.min() + 1e-10)

        # Plot epoch 1 (top row)
        ax_epoch1 = fig.add_subplot(gs[0, col_idx])
        ax_epoch1.imshow(sample_image, cmap='gray', alpha=0.7)
        im_epoch1 = ax_epoch1.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
        ax_epoch1.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nEpoch 1', fontsize=10)
        ax_epoch1.axis('off')

        # Plot final epoch (bottom row)
        ax_final = fig.add_subplot(gs[1, col_idx])
        ax_final.imshow(sample_image, cmap='gray', alpha=0.7)
        im_final = ax_final.imshow(occ_final_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
        ax_final.set_title(f'{depth} layer{"s" if depth > 1 else ""}\nFinal Epoch', fontsize=10)
        ax_final.axis('off')

        found_any = True

    # Check if we found any data
    if not found_any:
        plt.close(fig)
        print("No nopixel occlusion data found for any target depths. Skipping nopixel occlusion plot.")
        print("Run experiments with new code to generate nopixel occlusion data.")
        return

    # Add colorbar
    fig.colorbar(im_final, ax=fig.get_axes(), orientation='vertical',
                 label='Occlusion Sensitivity (normalized)', shrink=0.6, pad=0.02)

    fig.suptitle('Experiment 4: Occlusion Sensitivity Averaged Across All Digits (No Special Pixel)',
                 fontsize=16, y=0.98)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_4_occlusion_sensitivity_nopixel.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 4 nopixel occlusion plot saved to plots/experiment_4_occlusion_sensitivity_nopixel.png")


def plot_experiment_6_occlusion_multi_epoch():
    """Plot Experiment 6 occlusion sensitivity across multiple epochs (5 epochs × 5 architectures)."""
    import matplotlib.gridspec as gridspec
    import glob

    results_dir = 'results/exp6'

    # Full-depth VGG architectures: VGG9, VGG11, VGG13, VGG16, VGG19
    architectures = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
    arch_display_names = ['VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19']
    epochs = [1, 2, 3, 4, 5]

    # Create figure with 5 rows (epochs) × 5 columns (architectures)
    fig = plt.figure(figsize=(5 * len(architectures), 4 * len(epochs)))
    gs = gridspec.GridSpec(len(epochs), len(architectures), figure=fig, hspace=0.25, wspace=0.15)

    found_any = False
    im = None

    # Iterate through epochs (rows) and architectures (columns)
    for row_idx, epoch in enumerate(epochs):
        for col_idx, (arch_name, display_name) in enumerate(zip(architectures, arch_display_names)):
            # Load all seeds for this architecture (5-epoch results)
            all_files = glob.glob(os.path.join(results_dir, f"{arch_name}_seed*_5epochs_results.npz"))

            if not all_files:
                print(f"No 5-epoch results found for {arch_name}")
                continue

            # Collect occlusion maps across all seeds AND all classes for this epoch
            epoch_maps_all_classes = []
            sample_image = None

            for result_file in all_files:
                data = np.load(result_file, allow_pickle=True)

                # Check if occlusion data exists for this epoch
                occlusion_key = f'occlusion_maps_epoch{epoch}'
                if occlusion_key not in data:
                    continue

                occlusion_map = data[occlusion_key]  # Shape: (10, 32, 32)

                # Average across all 10 classes for this seed
                epoch_maps_all_classes.append(np.mean(occlusion_map, axis=0))

                # Use seed 0's first sample image (class 0) just for display
                if sample_image is None and 'seed0' in result_file:
                    sample_image_key = f'sample_images_epoch{epoch}'
                    if sample_image_key in data:
                        sample_image = data[sample_image_key][0]

            if not epoch_maps_all_classes:
                print(f"No occlusion data found for {arch_name} epoch {epoch}")
                continue

            # Use first available sample image if seed0 not found
            if sample_image is None:
                data = np.load(all_files[0], allow_pickle=True)
                sample_image_key = f'sample_images_epoch{epoch}'
                if sample_image_key in data:
                    sample_image = data[sample_image_key][0]

            # Average occlusion maps across all seeds (already averaged across classes)
            occ_map = np.mean(epoch_maps_all_classes, axis=0)

            # Apply power transform to compress dynamic range
            power = 0.5  # Square root transform

            # Shift to non-negative before power transform
            occ_shifted = occ_map - occ_map.min()
            occ_transformed = occ_shifted ** power
            occ_norm = (occ_transformed - occ_transformed.min()) / (occ_transformed.max() - occ_transformed.min() + 1e-10)

            if sample_image is not None:
                # Convert RGB image from (3, 32, 32) to (32, 32, 3) for display
                sample_image_display = np.transpose(sample_image, (1, 2, 0))
                # Normalize to [0, 1] for display
                sample_image_display = (sample_image_display - sample_image_display.min()) / (sample_image_display.max() - sample_image_display.min() + 1e-10)

            # Plot this epoch × architecture cell
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if sample_image is not None:
                ax.imshow(sample_image_display)
            im = ax.imshow(occ_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)

            # Add title only for top row (architecture names) and left column (epoch numbers)
            if row_idx == 0:
                ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Epoch {epoch}', fontsize=11, fontweight='bold')

            ax.axis('off')

            found_any = True

    # Check if we found any data
    if not found_any:
        plt.close(fig)
        print(f"No multi-epoch occlusion data found for experiment 6")
        return

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Occlusion Sensitivity (normalized)', rotation=270, labelpad=20, fontsize=11)

    fig.suptitle('Experiment 6: Full-Depth VGG Occlusion Sensitivity Across Training (Averaged Across All Classes)',
                 fontsize=15, fontweight='bold', y=0.995)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_6_occlusion_sensitivity_multi_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 6 multi-epoch occlusion plot saved to plots/experiment_6_occlusion_sensitivity_multi_epoch.png")


def plot_experiment_6_occlusion():
    """Plot Experiment 6 occlusion sensitivity (epoch 1 only, full-depth VGGs)."""
    import matplotlib.gridspec as gridspec
    import glob

    results_dir = 'results/exp6'

    # Full-depth VGG architectures: VGG9, VGG11, VGG13, VGG16, VGG19
    architectures = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
    arch_display_names = ['VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19']

    # Create figure with single row
    fig = plt.figure(figsize=(5 * len(architectures), 5))
    gs = gridspec.GridSpec(1, len(architectures), figure=fig, hspace=0.3, wspace=0.3)

    found_any = False
    im = None

    for col_idx, (arch_name, display_name) in enumerate(zip(architectures, arch_display_names)):
        # Load all seeds and average occlusion maps
        all_files = glob.glob(os.path.join(results_dir, f"{arch_name}_seed*_results.npz"))

        if not all_files:
            print(f"No results found for {arch_name}")
            continue

        # Collect occlusion maps across all seeds AND all classes
        epoch1_maps_all_classes = []
        sample_image = None

        for result_file in all_files:
            data = np.load(result_file, allow_pickle=True)

            # Check if occlusion data exists
            if 'occlusion_maps_epoch1' not in data:
                continue

            occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 32, 32)

            # Average across all 10 classes for this seed
            epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))

            # Use seed 0's first sample image (class 0) just for display
            if sample_image is None and 'seed0' in result_file:
                sample_image = data['sample_images_epoch1'][0]

        if not epoch1_maps_all_classes:
            print(f"No occlusion data found for {arch_name}")
            continue

        # Use first available sample image if seed0 not found
        if sample_image is None:
            data = np.load(all_files[0], allow_pickle=True)
            if 'sample_images_epoch1' in data:
                sample_image = data['sample_images_epoch1'][0]

        # Average occlusion maps across all seeds (already averaged across classes)
        occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)

        # Apply power transform to compress dynamic range
        power = 0.5  # Square root transform

        # Shift to non-negative before power transform
        occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
        occ_epoch1_transformed = occ_epoch1_shifted ** power
        occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

        if sample_image is not None:
            # Convert RGB image from (3, 32, 32) to (32, 32, 3) for display
            sample_image_display = np.transpose(sample_image, (1, 2, 0))
            # Normalize to [0, 1] for display
            sample_image_display = (sample_image_display - sample_image_display.min()) / (sample_image_display.max() - sample_image_display.min() + 1e-10)

        # Plot epoch 1
        ax = fig.add_subplot(gs[0, col_idx])
        if sample_image is not None:
            ax.imshow(sample_image_display)
        im = ax.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)
        ax.set_title(f'{display_name}\nEpoch 1', fontsize=12)
        ax.axis('off')

        found_any = True

    # Check if we found any data
    if not found_any:
        plt.close(fig)
        print(f"No occlusion data found for experiment 6")
        return

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Occlusion Sensitivity (normalized)', rotation=270, labelpad=20)

    fig.suptitle('Experiment 6: Full-Depth VGG Occlusion Sensitivity at Epoch 1 (Averaged Across All Classes)',
                 fontsize=14, y=0.98)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/experiment_6_occlusion_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Experiment 6 occlusion plot saved to plots/experiment_6_occlusion_sensitivity.png")


def main():
    print("Generating plots...")
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    plot_experiment_3_occlusion()
    plot_experiment_3_occlusion_nopixel()
    plot_experiment_4()
    plot_experiment_4_occlusion()
    plot_experiment_4_occlusion_nopixel()
    plot_experiment_5()
    plot_experiment_6_occlusion()
    print("All plots generated successfully!")


def plot_combined_mlp_vgg_occlusion():
    """
    Create a combined 2-row figure showing occlusion sensitivity:
    - Row 1: MLPs (5, 10, 15, 20, 25 layers) at Epoch 1
    - Row 2: VGGs (VGG9, VGG11, VGG13, VGG16, VGG19) at Epoch 1
    Publication-quality styling with Times New Roman font.
    """
    import matplotlib.gridspec as gridspec
    import glob

    # Configuration
    mlp_depths = [5, 10, 15, 20, 25]
    vgg_archs = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
    vgg_labels = ['VGG9', 'VGG11', 'VGG13', 'VGG16', 'VGG19']

    # Create figure with 2 rows × 5 columns
    fig = plt.figure(figsize=(16, 6.5))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.15, wspace=0.01,
                          left=0.05, right=0.92, top=0.95, bottom=0.08)

    im = None  # Will store the last imshow object for colorbar

    # ==================== ROW 1: MLPs ====================
    for col_idx, depth in enumerate(mlp_depths):
        # Load all seeds and average occlusion maps
        all_files = glob.glob(os.path.join('results/exp4', f"mlp_layers{depth}_seed*_results.npz"))
        matching_files = [f for f in all_files if 'nopixel' not in f]

        if not matching_files:
            print(f"No results found for MLP {depth} layers")
            continue

        # Collect occlusion maps across all seeds and classes
        epoch1_maps_all_classes = []
        sample_image = None

        for result_file in matching_files:
            data = np.load(result_file, allow_pickle=True)

            if 'occlusion_maps_epoch1' not in data:
                continue

            occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 28, 28)

            # Average across all 10 classes for this seed
            epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))

            # Use seed 0's first sample image for display
            if sample_image is None and 'seed0' in result_file:
                sample_image = data['sample_images_epoch1'][0]

        if not epoch1_maps_all_classes:
            print(f"No occlusion data found for MLP {depth} layers")
            continue

        # Use first available sample image if seed0 not found
        if sample_image is None:
            data = np.load(matching_files[0], allow_pickle=True)
            sample_image = data['sample_images_epoch1'][0]

        # Average occlusion maps across all seeds
        occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)

        # Apply power transform and normalize
        power = 0.5  # Square root transform
        occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
        occ_epoch1_transformed = occ_epoch1_shifted ** power
        occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / \
                          (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

        # Plot MLP occlusion (row 0)
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(sample_image, cmap='gray', alpha=0.7)
        im = ax.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)

        # Title with row label only on leftmost column
        if col_idx == 0:
            ax.set_title(f'MLP: {depth} layers', fontsize=16, fontweight='bold', pad=10)
        else:
            ax.set_title(f'{depth} layers', fontsize=16, fontweight='bold', pad=10)

        ax.axis('off')

    # ==================== ROW 2: VGGs ====================
    for col_idx, (arch_name, display_name) in enumerate(zip(vgg_archs, vgg_labels)):
        # Load all seeds and average occlusion maps
        all_files = glob.glob(os.path.join('results/exp6', f"{arch_name}_seed*_results.npz"))

        if not all_files:
            print(f"No results found for {arch_name}")
            continue

        # Collect occlusion maps across all seeds and classes
        epoch1_maps_all_classes = []
        sample_image = None

        for result_file in all_files:
            data = np.load(result_file, allow_pickle=True)

            if 'occlusion_maps_epoch1' not in data:
                continue

            occlusion_epoch1 = data['occlusion_maps_epoch1']  # Shape: (10, 32, 32)

            # Average across all 10 classes for this seed
            epoch1_maps_all_classes.append(np.mean(occlusion_epoch1, axis=0))

            # Use seed 0's first sample image for display
            if sample_image is None and 'seed0' in result_file:
                sample_image = data['sample_images_epoch1'][0]

        if not epoch1_maps_all_classes:
            print(f"No occlusion data found for {arch_name}")
            continue

        # Use first available sample image if seed0 not found
        if sample_image is None:
            data = np.load(all_files[0], allow_pickle=True)
            if 'sample_images_epoch1' in data:
                sample_image = data['sample_images_epoch1'][0]

        # Average occlusion maps across all seeds
        occ_map_epoch1 = np.mean(epoch1_maps_all_classes, axis=0)

        # Apply power transform and normalize
        power = 0.5
        occ_epoch1_shifted = occ_map_epoch1 - occ_map_epoch1.min()
        occ_epoch1_transformed = occ_epoch1_shifted ** power
        occ_epoch1_norm = (occ_epoch1_transformed - occ_epoch1_transformed.min()) / \
                          (occ_epoch1_transformed.max() - occ_epoch1_transformed.min() + 1e-10)

        if sample_image is not None:
            # Convert RGB image from (3, 32, 32) to (32, 32, 3) for display
            sample_image_display = np.transpose(sample_image, (1, 2, 0))
            # Normalize to [0, 1] for display
            sample_image_display = (sample_image_display - sample_image_display.min()) / \
                                   (sample_image_display.max() - sample_image_display.min() + 1e-10)

        # Plot VGG occlusion (row 1)
        ax = fig.add_subplot(gs[1, col_idx])
        if sample_image is not None:
            ax.imshow(sample_image_display)
        im = ax.imshow(occ_epoch1_norm, cmap='hot', alpha=1.0, vmin=0, vmax=1)

        # Title with row label only on leftmost column
        if col_idx == 0:
            ax.set_title(f'CNN: {display_name}', fontsize=16, fontweight='bold', pad=10)
        else:
            ax.set_title(display_name, fontsize=16, fontweight='bold', pad=10)

        ax.axis('off')

    # Add colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Occlusion Sensitivity', rotation=270, labelpad=25,
                      fontsize=16, fontweight='normal')
        cbar.ax.tick_params(labelsize=14)

    # Save figure
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/combined_mlp_vgg_occlusion.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("Combined MLP-VGG occlusion plot saved to plots/combined_mlp_vgg_occlusion.png")


if __name__ == "__main__":
    main()