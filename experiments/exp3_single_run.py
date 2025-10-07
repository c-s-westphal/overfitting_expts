import torch
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preact_resnet_cifar import build_preact_resnet
from models.vgg_cifar import VGG7, VGG9, VGG11, VGG13, VGG16, VGG19
from data.data_loader import get_cifar10_special_pixel_dataloaders
from utils.training import train_model


def noise_for_target_mi(target_mi_bits, num_classes=10):
    # Inverse of mi = log2(C) * (1 - 2*noise)
    max_mi = np.log2(num_classes)
    noise = (1.0 - (target_mi_bits / max_mi)) / 2.0
    return float(noise)


def main():
    parser = argparse.ArgumentParser(description='Single run for Experiment 3: depth sweep at fixed MI')
    parser.add_argument('--arch', type=str, default='preactresnet', choices=['preactresnet', 'vgg'], help='Model architecture')
    parser.add_argument('--depth', type=int, required=True, help='Depth spec. For PreActResNet: total layers (e.g., 20, 32, 44, 56, 80, 110, 218). For VGG: {7, 9, 11, 13, 16, 19}.')
    parser.add_argument('--mi_bits', type=float, default=2.5, help='Target MI in bits for special pixel (default: 2.5)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log_dir', type=str, required=False, help='Unused placeholder for symmetry with exp2')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    noise_level = noise_for_target_mi(args.mi_bits, num_classes=10)
    if args.arch == 'preactresnet':
        print(f"Running: PreActResNet depth={args.depth}, noise={noise_level:.4f} (MI={args.mi_bits} bits), seed={args.seed}")
        # prefer_stable=True selects bottleneck automatically for valid deep depths (>=164, 9n+2)
        model = build_preact_resnet(args.depth, num_classes=10, prefer_stable=True)
        model_label = 'PreActResNet'
        save_prefix = 'preactresnet'
    else:
        print(f"Running: VGG depth={args.depth}, noise={noise_level:.4f} (MI={args.mi_bits} bits), seed={args.seed}")
        vgg_map = {
            7: VGG7,
            9: VGG9,
            11: VGG11,
            13: VGG13,
            16: VGG16,
            19: VGG19,
        }
        if args.depth not in vgg_map:
            raise ValueError(f"Unsupported VGG depth: {args.depth}. Allowed: {sorted(vgg_map.keys())}")
        model = vgg_map[args.depth]()
        model_label = 'VGG'
        save_prefix = 'vgg'

    trainloader, testloader = get_cifar10_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=noise_level,
        seed=args.seed
    )

    metrics = train_model(
        model,
        trainloader,
        testloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    if metrics['final_train_acc'] >= 99.0:
        result = {
            'model': model_label,
            'depth': args.depth,
            'noise_level': noise_level,
            'mi_bits': args.mi_bits,
            'seed': args.seed,
            'train_acc': metrics['final_train_acc'],
            'test_acc': metrics['final_test_acc'],
            'generalization_gap': metrics['generalization_gap'],
            'train_loss': metrics['final_train_loss'],
            'test_loss': metrics['final_test_loss'],
            'valid': True
        }

        os.makedirs(args.output_dir, exist_ok=True)
        save_path = f"{args.output_dir}/{save_prefix}_depth{args.depth}_seed{args.seed}_results.npz"
        np.savez(save_path, **result)
        print(f"Results saved to {save_path}")
        print(f"Final train acc: {metrics['final_train_acc']:.2f}%, test acc: {metrics['final_test_acc']:.2f}%")
    else:
        print(f"Warning: {model_label} depth {args.depth} achieved only {metrics['final_train_acc']:.2f}% train accuracy")
        result = {
            'model': model_label,
            'depth': args.depth,
            'noise_level': noise_level,
            'mi_bits': args.mi_bits,
            'seed': args.seed,
            'valid': False
        }
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = f"{args.output_dir}/{save_prefix}_depth{args.depth}_seed{args.seed}_results.npz"
        np.savez(save_path, **result)


if __name__ == "__main__":
    main()


