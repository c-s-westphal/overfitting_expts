import numpy as np
import os
import re
from utils.mi_calculation import calculate_mutual_information


def load_experiment_results(experiment_type, model_name):
    if experiment_type == 'exp1':
        path = f'results/exp1/{model_name}_dataset_size_results.npz'
    elif experiment_type == 'exp2':
        path = f'results/exp2/{model_name}_special_pixel_results.npz'
    elif experiment_type == 'exp3':
        path = f'results/exp3/{model_name}_depth_sweep_results.npz'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    
    return np.load(path, allow_pickle=True)


def aggregate_results(results_dict):
    aggregated = {}

    for key in ['train_accs', 'test_accs', 'generalization_gaps', 'train_losses', 'test_losses', 'epochs_to_100pct', 'epochs_to_99pct']:
        if key in results_dict:
            data = results_dict[key]
            means = []
            stds = []

            for size_results in data:
                valid_results = [r for r in size_results if r is not None]
                if valid_results:
                    means.append(np.mean(valid_results))
                    stds.append(np.std(valid_results))
                else:
                    means.append(None)
                    stds.append(None)

            aggregated[f'{key}_mean'] = means
            aggregated[f'{key}_std'] = stds

    aggregated['valid_counts'] = []
    for valid_list in results_dict['valid_results']:
        aggregated['valid_counts'].append(sum(valid_list))

    return aggregated


def print_summary(experiment_type, model_name):
    results = load_experiment_results(experiment_type, model_name)
    aggregated = aggregate_results(results)
    
    print(f"\n{model_name} - Experiment {experiment_type}")
    print("=" * 50)
    
    if experiment_type == 'exp1':
        sizes = results['dataset_sizes']
        for i, size in enumerate(sizes):
            valid = aggregated['valid_counts'][i]
            if aggregated['generalization_gaps_mean'][i] is not None:
                print(f"Size {size:6d}: Gap = {aggregated['generalization_gaps_mean'][i]:.2f} ± {aggregated['generalization_gaps_std'][i]:.2f} (valid: {valid}/5)")
    elif experiment_type == 'exp2':
        noise_levels = results['noise_levels']
        mi_values = results['mi_values']
        for i, (noise, mi) in enumerate(zip(noise_levels, mi_values)):
            valid = aggregated['valid_counts'][i]
            if aggregated['generalization_gaps_mean'][i] is not None:
                print(f"Noise {noise:.3f} (MI={mi:.3f}): Gap = {aggregated['generalization_gaps_mean'][i]:.2f} ± {aggregated['generalization_gaps_std'][i]:.2f} (valid: {valid}/5)")
    elif experiment_type == 'exp3':
        # Handle both old 'depths' and new 'n_layers' keys
        layer_key = 'n_layers' if 'n_layers' in results else 'depths'
        layers = results[layer_key]
        for i, layer in enumerate(layers):
            valid = aggregated['valid_counts'][i]
            if aggregated['generalization_gaps_mean'][i] is not None:
                print(f"Layers {layer:4d}: Gap = {aggregated['generalization_gaps_mean'][i]:.2f} ± {aggregated['generalization_gaps_std'][i]:.2f} (valid: {valid}/3)")


def load_exp1_results_from_per_seed(model_name, results_dir='results/exp1'):
    """Aggregate per-seed exp1 results on the fly for a given model.

    Expects files named like: "{model_lower}_size{size}_seed{seed}_results.npz".
    Returns a dict compatible with aggregate_results.
    """
    model_key = model_name.lower()
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    file_pattern = re.compile(rf"^{re.escape(model_key)}_size(\d+)_seed(\d+)_results\.npz$")

    size_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        size = int(match.group(1))
        seed = int(match.group(2))
        size_to_seeds.setdefault(size, set()).add(seed)

    if not size_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for {model_name} in {results_dir}"
        )

    dataset_sizes = sorted(size_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in size_to_seeds.values() for s in seeds})

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    valid_results = []

    for size in dataset_sizes:
        size_train_accs = []
        size_test_accs = []
        size_gen_gaps = []
        size_train_losses = []
        size_test_losses = []
        size_valid = []

        seeds_for_size = sorted(size_to_seeds[size])
        for seed in seeds_for_size:
            fpath = os.path.join(
                results_dir, f"{model_key}_size{size}_seed{seed}_results.npz"
            )
            if not os.path.exists(fpath):
                # skip silently per user request
                continue
            data = np.load(fpath, allow_pickle=True)
            is_valid = bool(data.get('valid', False))
            if is_valid:
                size_train_accs.append(float(data['train_acc']))
                size_test_accs.append(float(data['test_acc']))
                size_gen_gaps.append(float(data['generalization_gap']))
                size_train_losses.append(float(data['train_loss']))
                size_test_losses.append(float(data['test_loss']))
                size_valid.append(True)
            else:
                size_train_accs.append(None)
                size_test_accs.append(None)
                size_gen_gaps.append(None)
                size_train_losses.append(None)
                size_test_losses.append(None)
                size_valid.append(False)

        train_accs.append(size_train_accs)
        test_accs.append(size_test_accs)
        generalization_gaps.append(size_gen_gaps)
        train_losses.append(size_train_losses)
        test_losses.append(size_test_losses)
        valid_results.append(size_valid)

    return {
        'model': model_name,
        'dataset_sizes': dataset_sizes,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'valid_results': valid_results,
    }


def load_exp2_results_from_per_seed(model_name, results_dir='results/exp2'):
    """Aggregate per-seed exp2 results on the fly for a given model.

    Expects files named like: "{model_lower}_noise{noise:.3f}_seed{seed}_results.npz".
    Returns a dict with 'noise_levels' and 'mi_values' to match the aggregator expectations.
    """
    model_key = model_name.lower()
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    file_pattern = re.compile(rf"^{re.escape(model_key)}_noise(\d+\.\d+)_seed(\d+)_results\.npz$")

    noise_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        noise_str = match.group(1)
        seed = int(match.group(2))
        noise_level = float(noise_str)
        noise_to_seeds.setdefault(noise_level, set()).add(seed)

    if not noise_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for {model_name} in {results_dir}"
        )

    noise_levels = sorted(noise_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in noise_to_seeds.values() for s in seeds})

    # Compute MI per noise level (same formula used elsewhere)
    mi_values = [calculate_mutual_information(n) for n in noise_levels]

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    valid_results = []

    for noise in noise_levels:
        noise_train_accs = []
        noise_test_accs = []
        noise_gen_gaps = []
        noise_train_losses = []
        noise_test_losses = []
        noise_valid = []

        seeds_for_noise = sorted(noise_to_seeds[noise])
        for seed in seeds_for_noise:
            fpath = os.path.join(
                results_dir, f"{model_key}_noise{noise:.3f}_seed{seed}_results.npz"
            )
            if not os.path.exists(fpath):
                # skip silently per user request
                continue
            data = np.load(fpath, allow_pickle=True)
            is_valid = bool(data.get('valid', False))
            if is_valid:
                noise_train_accs.append(float(data['train_acc']))
                noise_test_accs.append(float(data['test_acc']))
                noise_gen_gaps.append(float(data['generalization_gap']))
                noise_train_losses.append(float(data['train_loss']))
                noise_test_losses.append(float(data['test_loss']))
                noise_valid.append(True)
            else:
                noise_train_accs.append(None)
                noise_test_accs.append(None)
                noise_gen_gaps.append(None)
                noise_train_losses.append(None)
                noise_test_losses.append(None)
                noise_valid.append(False)

        train_accs.append(noise_train_accs)
        test_accs.append(noise_test_accs)
        generalization_gaps.append(noise_gen_gaps)
        train_losses.append(noise_train_losses)
        test_losses.append(noise_test_losses)
        valid_results.append(noise_valid)

    return {
        'model': model_name,
        'noise_levels': noise_levels,
        'mi_values': mi_values,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'valid_results': valid_results,
    }


def load_exp3_results_from_per_seed(model_name, results_dir='results/exp3', include_nopixel=False):
    """Aggregate per-seed exp3 results on the fly for a given model.

    Expects files named like: "vgg{X}var_layers{n}_seed{seed}_results.npz".
    If include_nopixel=True, also loads "vgg{X}var_layers{n}_seed{seed}_nopixel_results.npz".
    Returns a dict compatible with aggregate_results, including 'n_layers'.
    """
    # Map model_name to file prefix
    # model_name should be like 'VGG11', 'VGG13', 'VGG16', 'VGG19'
    model_num = model_name.replace('VGG', '').lower()
    file_prefix = f"vgg{model_num}var"

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    suffix = "_nopixel" if include_nopixel else ""
    file_pattern = re.compile(rf"^{re.escape(file_prefix)}_layers(\d+)_seed(\d+){re.escape(suffix)}_results\.npz$")

    layers_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        n_layers = int(match.group(1))
        seed = int(match.group(2))
        layers_to_seeds.setdefault(n_layers, set()).add(seed)

    if not layers_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for {model_name} in {results_dir}"
        )

    n_layers_list = sorted(layers_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in layers_to_seeds.values() for s in seeds})

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    epochs_to_100pct = []
    valid_results = []

    for n_layers in n_layers_list:
        layers_train_accs = []
        layers_test_accs = []
        layers_gen_gaps = []
        layers_train_losses = []
        layers_test_losses = []
        layers_epochs_to_100pct = []
        layers_valid = []

        seeds_for_layers = sorted(layers_to_seeds[n_layers])
        for seed in seeds_for_layers:
            fpath = os.path.join(
                results_dir, f"{file_prefix}_layers{n_layers}_seed{seed}{suffix}_results.npz"
            )
            if not os.path.exists(fpath):
                # skip silently per user request
                continue
            data = np.load(fpath, allow_pickle=True)
            is_valid = bool(data.get('valid', False))
            if is_valid:
                layers_train_accs.append(float(data['train_acc']))
                layers_test_accs.append(float(data['test_acc']))
                layers_gen_gaps.append(float(data['generalization_gap']))
                layers_train_losses.append(float(data['train_loss']))
                layers_test_losses.append(float(data['test_loss']))

                # Load epochs_to_100pct if available
                if 'epochs_to_100pct' in data:
                    epochs_val = int(data['epochs_to_100pct'])
                    layers_epochs_to_100pct.append(epochs_val if epochs_val != -1 else None)
                else:
                    layers_epochs_to_100pct.append(None)

                layers_valid.append(True)
            else:
                layers_train_accs.append(None)
                layers_test_accs.append(None)
                layers_gen_gaps.append(None)
                layers_train_losses.append(None)
                layers_test_losses.append(None)
                layers_epochs_to_100pct.append(None)
                layers_valid.append(False)

        train_accs.append(layers_train_accs)
        test_accs.append(layers_test_accs)
        generalization_gaps.append(layers_gen_gaps)
        train_losses.append(layers_train_losses)
        test_losses.append(layers_test_losses)
        epochs_to_100pct.append(layers_epochs_to_100pct)
        valid_results.append(layers_valid)

    return {
        'model': model_name + (' (No Pixel)' if include_nopixel else ''),
        'n_layers': n_layers_list,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_to_100pct': epochs_to_100pct,
        'valid_results': valid_results,
    }


def load_exp4_results_from_per_seed(results_dir='results/exp4', include_nopixel=False):
    """Aggregate per-seed exp4 results on the fly for MLP Variable.

    Expects files named like: "mlp_layers{n}_seed{seed}_results.npz".
    If include_nopixel=True, also loads "mlp_layers{n}_seed{seed}_nopixel_results.npz".
    Returns a dict compatible with aggregate_results, including 'n_layers'.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    suffix = "_nopixel" if include_nopixel else ""
    file_pattern = re.compile(rf"^mlp_layers(\d+)_seed(\d+){re.escape(suffix)}_results\.npz$")

    layers_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        n_layers = int(match.group(1))
        seed = int(match.group(2))
        layers_to_seeds.setdefault(n_layers, set()).add(seed)

    if not layers_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for MLP{suffix} in {results_dir}"
        )

    n_layers_list = sorted(layers_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in layers_to_seeds.values() for s in seeds})

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    epochs_to_99pct = []
    valid_results = []

    for n_layers in n_layers_list:
        layers_train_accs = []
        layers_test_accs = []
        layers_gen_gaps = []
        layers_train_losses = []
        layers_test_losses = []
        layers_epochs_to_99pct = []
        layers_valid = []

        seeds_for_layers = sorted(layers_to_seeds[n_layers])
        for seed in seeds_for_layers:
            fpath = os.path.join(
                results_dir, f"mlp_layers{n_layers}_seed{seed}{suffix}_results.npz"
            )
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath, allow_pickle=True)

            # Skip files that don't have train_acc (old format)
            if 'train_acc' not in data.keys():
                continue

            is_valid = bool(data.get('valid', False))

            # Always append the values (even if not valid, since we changed the code to save them)
            layers_train_accs.append(float(data['train_acc']))
            layers_test_accs.append(float(data['test_acc']))
            layers_gen_gaps.append(float(data['generalization_gap']))
            layers_train_losses.append(float(data['train_loss']))
            layers_test_losses.append(float(data['test_loss']))

            # Load epochs_to_99pct if available
            if 'epochs_to_99pct' in data:
                epochs_val = int(data['epochs_to_99pct'])
                layers_epochs_to_99pct.append(epochs_val if epochs_val != -1 else None)
            else:
                layers_epochs_to_99pct.append(None)

            layers_valid.append(is_valid)

        train_accs.append(layers_train_accs)
        test_accs.append(layers_test_accs)
        generalization_gaps.append(layers_gen_gaps)
        train_losses.append(layers_train_losses)
        test_losses.append(layers_test_losses)
        epochs_to_99pct.append(layers_epochs_to_99pct)
        valid_results.append(layers_valid)

    return {
        'model': 'MLP' + (' (No Pixel)' if include_nopixel else ''),
        'n_layers': n_layers_list,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_to_99pct': epochs_to_99pct,
        'valid_results': valid_results,
    }


def load_exp5_results_from_per_seed(results_dir='results/exp5'):
    """
    Load exp5 (All-Conv Variable) results from per-seed .npz files.
    Expects files named like: "allconv_layers{n}_seed{seed}_results.npz".
    Returns a dict compatible with aggregate_results, including 'n_layers'.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    file_pattern = re.compile(r"^allconv_layers(\d+)_seed(\d+)_results\.npz$")

    layers_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        n_layers = int(match.group(1))
        seed = int(match.group(2))
        layers_to_seeds.setdefault(n_layers, set()).add(seed)

    if not layers_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for All-Conv in {results_dir}"
        )

    n_layers_list = sorted(layers_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in layers_to_seeds.values() for s in seeds})

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    valid_results = []

    for n_layers in n_layers_list:
        layers_train_accs = []
        layers_test_accs = []
        layers_gen_gaps = []
        layers_train_losses = []
        layers_test_losses = []
        layers_valid = []

        seeds_for_layers = sorted(layers_to_seeds[n_layers])
        for seed in seeds_for_layers:
            fpath = os.path.join(
                results_dir, f"allconv_layers{n_layers}_seed{seed}_results.npz"
            )
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath, allow_pickle=True)

            # Skip files that don't have train_acc (old format)
            if 'train_acc' not in data.keys():
                continue

            is_valid = bool(data.get('valid', False))

            # Always append the values (even if not valid, since we changed the code to save them)
            layers_train_accs.append(float(data['train_acc']))
            layers_test_accs.append(float(data['test_acc']))
            layers_gen_gaps.append(float(data['generalization_gap']))
            layers_train_losses.append(float(data['train_loss']))
            layers_test_losses.append(float(data['test_loss']))
            layers_valid.append(is_valid)

        train_accs.append(layers_train_accs)
        test_accs.append(layers_test_accs)
        generalization_gaps.append(layers_gen_gaps)
        train_losses.append(layers_train_losses)
        test_losses.append(layers_test_losses)
        valid_results.append(layers_valid)

    return {
        'model': 'AllConv',
        'n_layers': n_layers_list,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'valid_results': valid_results,
    }