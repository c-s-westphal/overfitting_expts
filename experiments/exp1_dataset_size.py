import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_cifar import ResNet20, ResNet32, ResNet56
from models.vgg_cifar import VGG11, VGG16, VGG19
from data.data_loader import get_cifar10_dataloaders
from utils.training import train_model
from tqdm import tqdm


def run_dataset_size_experiment(model_name, model_fn, dataset_sizes, seeds, device='cuda'):
    results = {
        'model': model_name,
        'dataset_sizes': dataset_sizes,
        'seeds': seeds,
        'train_accs': [],
        'test_accs': [],
        'generalization_gaps': [],
        'train_losses': [],
        'test_losses': [],
        'valid_results': []
    }
    
    for size in tqdm(dataset_sizes, desc=f"Dataset sizes for {model_name}"):
        size_train_accs = []
        size_test_accs = []
        size_gen_gaps = []
        size_train_losses = []
        size_test_losses = []
        size_valid = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = model_fn()
            
            trainloader, testloader = get_cifar10_dataloaders(
                batch_size=128,
                num_workers=4,
                subset_size=size if size < 60000 else None,
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
                size_train_accs.append(metrics['final_train_acc'])
                size_test_accs.append(metrics['final_test_acc'])
                size_gen_gaps.append(metrics['generalization_gap'])
                size_train_losses.append(metrics['final_train_loss'])
                size_test_losses.append(metrics['final_test_loss'])
                size_valid.append(True)
            else:
                size_train_accs.append(None)
                size_test_accs.append(None)
                size_gen_gaps.append(None)
                size_train_losses.append(None)
                size_test_losses.append(None)
                size_valid.append(False)
                print(f"Warning: {model_name} with size {size}, seed {seed} achieved only {metrics['final_train_acc']:.2f}% train accuracy")
        
        results['train_accs'].append(size_train_accs)
        results['test_accs'].append(size_test_accs)
        results['generalization_gaps'].append(size_gen_gaps)
        results['train_losses'].append(size_train_losses)
        results['test_losses'].append(size_test_losses)
        results['valid_results'].append(size_valid)
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset_sizes = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]
    seeds = [0, 1, 2]
    
    models = {
        'ResNet20': ResNet20,
        'ResNet32': ResNet32,
        'ResNet56': ResNet56,
        'VGG11': VGG11,
        'VGG16': VGG16,
        'VGG19': VGG19
    }
    
    os.makedirs('results/exp1', exist_ok=True)
    
    for model_name, model_fn in models.items():
        print(f"\nRunning experiments for {model_name}")
        results = run_dataset_size_experiment(
            model_name, 
            model_fn,
            dataset_sizes, 
            seeds,
            device
        )
        
        save_path = f'results/exp1/{model_name}_dataset_size_results.npz'
        np.savez(save_path, **results)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()