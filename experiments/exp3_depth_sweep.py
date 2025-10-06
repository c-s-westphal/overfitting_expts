import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preact_resnet_cifar import build_preact_resnet
from data.data_loader import get_cifar10_special_pixel_dataloaders
from utils.training import train_model


def noise_for_target_mi(target_mi_bits, num_classes=10):
    max_mi = np.log2(num_classes)
    return float((1.0 - (target_mi_bits / max_mi)) / 2.0)


def run_depth_sweep(depths, seeds, mi_bits=2.5, device='cuda'):
    noise_level = noise_for_target_mi(mi_bits, num_classes=10)

    results = {
        'model': 'PreActResNet',
        'depths': depths,
        'mi_bits': mi_bits,
        'noise_level': noise_level,
        'seeds': seeds,
        'train_accs': [],
        'test_accs': [],
        'generalization_gaps': [],
        'train_losses': [],
        'test_losses': [],
        'valid_results': []
    }

    for depth in depths:
        depth_train_accs = []
        depth_test_accs = []
        depth_gen_gaps = []
        depth_train_losses = []
        depth_test_losses = []
        depth_valid = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = build_preact_resnet(depth, num_classes=10, prefer_stable=True)

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
                depth_train_accs.append(metrics['final_train_acc'])
                depth_test_accs.append(metrics['final_test_acc'])
                depth_gen_gaps.append(metrics['generalization_gap'])
                depth_train_losses.append(metrics['final_train_loss'])
                depth_test_losses.append(metrics['final_test_loss'])
                depth_valid.append(True)
            else:
                depth_train_accs.append(None)
                depth_test_accs.append(None)
                depth_gen_gaps.append(None)
                depth_train_losses.append(None)
                depth_test_losses.append(None)
                depth_valid.append(False)
                print(f"Warning: depth {depth}, seed {seed} achieved only {metrics['final_train_acc']:.2f}% train accuracy")

        results['train_accs'].append(depth_train_accs)
        results['test_accs'].append(depth_test_accs)
        results['generalization_gaps'].append(depth_gen_gaps)
        results['train_losses'].append(depth_train_losses)
        results['test_losses'].append(depth_test_losses)
        results['valid_results'].append(depth_valid)

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    depths = [20, 32, 44, 56, 80, 110, 218]
    seeds = [0, 1, 2]

    print("Depths and MI setting:")
    print("  depths:", depths)
    print("  MI bits: 2.5 (noise fixed per formula), test set clean")

    os.makedirs('results/exp3', exist_ok=True)

    print("\nRunning experiments for PreActResNet")
    results = run_depth_sweep(depths, seeds, mi_bits=2.5, device=device)
    save_path = f'results/exp3/PreActResNet_depth_sweep_results.npz'
    np.savez(save_path, **results)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()


