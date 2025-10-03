import numpy as np
import os


def load_experiment_results(experiment_type, model_name):
    if experiment_type == 'exp1':
        path = f'results/exp1/{model_name}_dataset_size_results.npz'
    elif experiment_type == 'exp2':
        path = f'results/exp2/{model_name}_special_pixel_results.npz'
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
    else:
        noise_levels = results['noise_levels']
        mi_values = results['mi_values']
        for i, (noise, mi) in enumerate(zip(noise_levels, mi_values)):
            valid = aggregated['valid_counts'][i]
            if aggregated['generalization_gaps_mean'][i] is not None:
                print(f"Noise {noise:.3f} (MI={mi:.3f}): Gap = {aggregated['generalization_gaps_mean'][i]:.2f} ± {aggregated['generalization_gaps_std'][i]:.2f} (valid: {valid}/5)")