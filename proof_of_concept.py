#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys

from models.resnet_cifar import ResNet20
from data.data_loader import get_cifar10_dataloaders
from utils.training import train_model


def run_proof_of_concept():
    print("Running Proof of Concept: ResNet20 on varying dataset sizes")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset_sizes = [1000, 2000]
    seeds = [0, 1]
    
    results = {
        'model': 'ResNet20',
        'dataset_sizes': dataset_sizes,
        'seeds': seeds,
        'train_accs': [],
        'test_accs': [],
        'generalization_gaps': [],
        'train_losses': [],
        'test_losses': [],
        'valid_results': []
    }
    
    os.makedirs('results/exp1', exist_ok=True)
    
    for size in dataset_sizes:
        print(f"\n--- Dataset size: {size} ---")
        
        size_train_accs = []
        size_test_accs = []
        size_gen_gaps = []
        size_train_losses = []
        size_test_losses = []
        size_valid = []
        
        for seed in seeds:
            print(f"\nSeed: {seed}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            model = ResNet20()
            
            trainloader, testloader = get_cifar10_dataloaders(
                batch_size=128,
                num_workers=2,
                subset_size=size,
                seed=seed
            )
            
            print(f"Training set size: {len(trainloader.dataset)}")
            print(f"Test set size: {len(testloader.dataset)}")
            
            # Run for only 1 epoch as proof of concept
            metrics = train_model(
                model, 
                trainloader, 
                testloader, 
                epochs=1,  # Only 1 epoch for POC
                lr=0.1,
                device=device
            )
            
            # Accept any training accuracy >= 10% for POC (very relaxed)
            min_train_acc = 10.0
            if metrics['final_train_acc'] >= min_train_acc:
                size_train_accs.append(metrics['final_train_acc'])
                size_test_accs.append(metrics['final_test_acc'])
                size_gen_gaps.append(metrics['generalization_gap'])
                size_train_losses.append(metrics['final_train_loss'])
                size_test_losses.append(metrics['final_test_loss'])
                size_valid.append(True)
                print(f"✅ Valid result: Train {metrics['final_train_acc']:.2f}%, Test {metrics['final_test_acc']:.2f}%, Gap {metrics['generalization_gap']:.2f}%")
            else:
                size_train_accs.append(None)
                size_test_accs.append(None)
                size_gen_gaps.append(None)
                size_train_losses.append(None)
                size_test_losses.append(None)
                size_valid.append(False)
                print(f"❌ Invalid result: Train accuracy {metrics['final_train_acc']:.2f}% < {min_train_acc}%")
        
        results['train_accs'].append(size_train_accs)
        results['test_accs'].append(size_test_accs)
        results['generalization_gaps'].append(size_gen_gaps)
        results['train_losses'].append(size_train_losses)
        results['test_losses'].append(size_test_losses)
        results['valid_results'].append(size_valid)
    
    # Save results in the same format as exp1 so plot_results.py can read it
    save_path = 'results/exp1/ResNet20_dataset_size_results.npz'
    np.savez(save_path, **results)
    print(f"\nResults saved to {save_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROOF OF CONCEPT SUMMARY")
    print("=" * 60)
    
    for i, size in enumerate(dataset_sizes):
        valid_count = sum(results['valid_results'][i])
        print(f"\nDataset size {size}:")
        print(f"  Valid runs: {valid_count}/{len(seeds)}")
        
        for j, seed in enumerate(seeds):
            if results['valid_results'][i][j]:
                train_acc = results['train_accs'][i][j]
                test_acc = results['test_accs'][i][j]
                gap = results['generalization_gaps'][i][j]
                print(f"  Seed {seed}: Train={train_acc:.1f}%, Test={test_acc:.1f}%, Gap={gap:.1f}%")
            else:
                print(f"  Seed {seed}: FAILED")
    
    print(f"\n🎉 Proof of concept completed!")
    print(f"📁 Results saved to: {save_path}")
    print(f"📊 You can now run: python plot_results.py")
    
    return results


if __name__ == "__main__":
    results = run_proof_of_concept()