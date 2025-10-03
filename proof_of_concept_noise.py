#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys

from models.resnet_cifar import ResNet20
from data.data_loader import get_cifar10_special_pixel_dataloaders
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim


def train_one_batch(model, dataloader, device='cuda'):
    """Train on just one batch for proof of concept"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Get just one batch
    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Train for one epoch on this single batch
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Evaluate on the same batch (train accuracy)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        train_loss = criterion(outputs, labels).item()
        _, predicted = outputs.max(1)
        train_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)

    return train_loss, train_acc


def evaluate_one_batch(model, dataloader, device='cuda'):
    """Evaluate on just one batch for proof of concept"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Get just one batch
    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels).item()
        _, predicted = outputs.max(1)
        acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)

    return loss, acc


def run_proof_of_concept():
    print("Running Proof of Concept: Noise Pixel vs Generalization Gap")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test different noise levels
    noise_levels = [0.0, 0.1, 0.3]
    seeds = [0, 1]

    results = {
        'model': 'ResNet20',
        'noise_levels': noise_levels,
        'seeds': seeds,
        'train_accs': [],
        'test_accs': [],
        'generalization_gaps': [],
        'train_losses': [],
        'test_losses': [],
        'valid_results': []
    }

    os.makedirs('results/exp2', exist_ok=True)

    for noise in noise_levels:
        print(f"\n--- Noise level: {noise} ---")

        noise_train_accs = []
        noise_test_accs = []
        noise_gen_gaps = []
        noise_train_losses = []
        noise_test_losses = []
        noise_valid = []

        for seed in seeds:
            print(f"\nSeed: {seed}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = ResNet20()

            trainloader, testloader = get_cifar10_special_pixel_dataloaders(
                batch_size=128,
                num_workers=2,
                noise_level=noise,
                seed=seed
            )

            print(f"Training set size: {len(trainloader.dataset)}")
            print(f"Test set size: {len(testloader.dataset)}")
            print(f"Using only 1 batch for POC")

            # Train on one batch only
            train_loss, train_acc = train_one_batch(model, trainloader, device=device)
            test_loss, test_acc = evaluate_one_batch(model, testloader, device=device)

            gen_gap = train_acc - test_acc

            # Very relaxed threshold for POC - just checking if it runs
            min_train_acc = 5.0
            if train_acc >= min_train_acc:
                noise_train_accs.append(train_acc)
                noise_test_accs.append(test_acc)
                noise_gen_gaps.append(gen_gap)
                noise_train_losses.append(train_loss)
                noise_test_losses.append(test_loss)
                noise_valid.append(True)
                print(f"✅ Valid result: Train {train_acc:.2f}%, Test {test_acc:.2f}%, Gap {gen_gap:.2f}%")
            else:
                noise_train_accs.append(None)
                noise_test_accs.append(None)
                noise_gen_gaps.append(None)
                noise_train_losses.append(None)
                noise_test_losses.append(None)
                noise_valid.append(False)
                print(f"❌ Invalid result: Train accuracy {train_acc:.2f}% < {min_train_acc}%")

        results['train_accs'].append(noise_train_accs)
        results['test_accs'].append(noise_test_accs)
        results['generalization_gaps'].append(noise_gen_gaps)
        results['train_losses'].append(noise_train_losses)
        results['test_losses'].append(noise_test_losses)
        results['valid_results'].append(noise_valid)

    # Save results
    save_path = 'results/exp2/ResNet20_noise_pixel_results.npz'
    np.savez(save_path, **results)
    print(f"\nResults saved to {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PROOF OF CONCEPT SUMMARY")
    print("=" * 60)

    for i, noise in enumerate(noise_levels):
        valid_count = sum(results['valid_results'][i])
        print(f"\nNoise level {noise}:")
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

    return results


if __name__ == "__main__":
    results = run_proof_of_concept()
