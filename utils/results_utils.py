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
    
    for key in ['train_accs', 'test_accs', 'generalization_gaps', 'train_losses', 'test_losses']:
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
        depths = results['depths']
        for i, depth in enumerate(depths):
            valid = aggregated['valid_counts'][i]
            if aggregated['generalization_gaps_mean'][i] is not None:
                print(f"Depth {depth:4d}: Gap = {aggregated['generalization_gaps_mean'][i]:.2f} ± {aggregated['generalization_gaps_std'][i]:.2f} (valid: {valid}/3)")


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


def load_exp3_results_from_per_seed(model_name, results_dir='results/exp3'):
    """Aggregate per-seed exp3 results on the fly for a given model.

    Expects files named like: "{model_lower}_depth{depth}_seed{seed}_results.npz".
    Returns a dict compatible with aggregate_results, including 'depths'.
    """
    model_key = model_name.lower()
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    file_pattern = re.compile(rf"^{re.escape(model_key)}_depth(\d+)_seed(\d+)_results\.npz$")

    depth_to_seeds = {}
    for fname in os.listdir(results_dir):
        match = file_pattern.match(fname)
        if not match:
            continue
        depth = int(match.group(1))
        seed = int(match.group(2))
        depth_to_seeds.setdefault(depth, set()).add(seed)

    if not depth_to_seeds:
        raise FileNotFoundError(
            f"No per-seed results found for {model_name} in {results_dir}"
        )

    depths = sorted(depth_to_seeds.keys())
    all_seeds_sorted = sorted({s for seeds in depth_to_seeds.values() for s in seeds})

    train_accs = []
    test_accs = []
    generalization_gaps = []
    train_losses = []
    test_losses = []
    valid_results = []

    for depth in depths:
        depth_train_accs = []
        depth_test_accs = []
        depth_gen_gaps = []
        depth_train_losses = []
        depth_test_losses = []
        depth_valid = []

        seeds_for_depth = sorted(depth_to_seeds[depth])
        for seed in seeds_for_depth:
            fpath = os.path.join(
                results_dir, f"{model_key}_depth{depth}_seed{seed}_results.npz"
            )
            if not os.path.exists(fpath):
                # skip silently per user request
                continue
            data = np.load(fpath, allow_pickle=True)
            is_valid = bool(data.get('valid', False))
            if is_valid:
                depth_train_accs.append(float(data['train_acc']))
                depth_test_accs.append(float(data['test_acc']))
                depth_gen_gaps.append(float(data['generalization_gap']))
                depth_train_losses.append(float(data['train_loss']))
                depth_test_losses.append(float(data['test_loss']))
                depth_valid.append(True)
            else:
                depth_train_accs.append(None)
                depth_test_accs.append(None)
                depth_gen_gaps.append(None)
                depth_train_losses.append(None)
                depth_test_losses.append(None)
                depth_valid.append(False)

        train_accs.append(depth_train_accs)
        test_accs.append(depth_test_accs)
        generalization_gaps.append(depth_gen_gaps)
        train_losses.append(depth_train_losses)
        test_losses.append(depth_test_losses)
        valid_results.append(depth_valid)

    return {
        'model': model_name,
        'depths': depths,
        'seeds': all_seeds_sorted,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'generalization_gaps': generalization_gaps,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'valid_results': valid_results,
    }