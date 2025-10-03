import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_cifar import ResNet20, ResNet32, ResNet56
from models.vgg_cifar import VGG11, VGG16, VGG19
from data.data_loader import get_cifar10_special_pixel_dataloaders
from utils.training import train_model
from utils.mi_calculation import get_noise_levels_with_mi
from tqdm import tqdm


def run_special_pixel_experiment(model_name, model_fn, noise_levels, mi_values, seeds, device='cuda'):
    results = {
        'model': model_name,
        'noise_levels': noise_levels,
        'mi_values': mi_values,
        'seeds': seeds,
        'train_accs': [],
        'test_accs': [],
        'generalization_gaps': [],
        'train_losses': [],
        'test_losses': [],
        'valid_results': []
    }
    
    for noise_level in tqdm(noise_levels, desc=f"Noise levels for {model_name}"):
        noise_train_accs = []
        noise_test_accs = []
        noise_gen_gaps = []
        noise_train_losses = []
        noise_test_losses = []
        noise_valid = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = model_fn()
            
            trainloader, testloader = get_cifar10_special_pixel_dataloaders(
                batch_size=128,
                num_workers=4,
                noise_level=noise_level,
                seed=seed
            )
            
            metrics = train_model(
                model, 
                trainloader, 
                testloader, 
                epochs=200,
                lr=0.1,
                device=device
            )
            
            if metrics['final_train_acc'] >= 99.0:
                noise_train_accs.append(metrics['final_train_acc'])
                noise_test_accs.append(metrics['final_test_acc'])
                noise_gen_gaps.append(metrics['generalization_gap'])
                noise_train_losses.append(metrics['final_train_loss'])
                noise_test_losses.append(metrics['final_test_loss'])
                noise_valid.append(True)
            else:
                noise_train_accs.append(None)
                noise_test_accs.append(None)
                noise_gen_gaps.append(None)
                noise_train_losses.append(None)
                noise_test_losses.append(None)
                noise_valid.append(False)
                print(f"Warning: {model_name} with noise {noise_level:.3f}, seed {seed} achieved only {metrics['final_train_acc']:.2f}% train accuracy")
        
        results['train_accs'].append(noise_train_accs)
        results['test_accs'].append(noise_test_accs)
        results['generalization_gaps'].append(noise_gen_gaps)
        results['train_losses'].append(noise_train_losses)
        results['test_losses'].append(noise_test_losses)
        results['valid_results'].append(noise_valid)
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    noise_levels, mi_values = get_noise_levels_with_mi(num_levels=10)
    seeds = [0, 1, 2]
    
    print("Noise levels and corresponding MI values:")
    for noise, mi in zip(noise_levels, mi_values):
        print(f"  Noise: {noise:.3f}, MI: {mi:.4f} bits")
    
    models = {
        'ResNet20': ResNet20,
        'ResNet32': ResNet32,
        'ResNet56': ResNet56,
        'VGG11': VGG11,
        'VGG16': VGG16,
        'VGG19': VGG19
    }
    
    os.makedirs('results/exp2', exist_ok=True)
    
    for model_name, model_fn in models.items():
        print(f"\nRunning experiments for {model_name}")
        results = run_special_pixel_experiment(
            model_name, 
            model_fn,
            noise_levels,
            mi_values,
            seeds,
            device
        )
        
        save_path = f'results/exp2/{model_name}_special_pixel_results.npz'
        np.savez(save_path, **results)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()