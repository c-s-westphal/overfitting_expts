import torch
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vgg_variable_family import VGG9_Variable, VGG11_Variable, VGG13_Variable, VGG16_Variable, VGG19_Variable
from data.data_loader import get_cifar10_special_pixel_dataloaders
from utils.training import train_model


def get_model_fn(model_name):
    """Get model function and full depth for VGG architectures."""
    models = {
        'vgg9': (VGG9_Variable, 6),
        'vgg11': (VGG11_Variable, 8),
        'vgg13': (VGG13_Variable, 10),
        'vgg16': (VGG16_Variable, 13),
        'vgg19': (VGG19_Variable, 16)
    }
    return models[model_name]


def main():
    parser = argparse.ArgumentParser(description='Single run for noise pixel experiment')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--noise_level', type=float, required=True, help='Noise level')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Running: {args.model}, noise={args.noise_level:.3f}, seed={args.seed}")

    model_fn, full_depth = get_model_fn(args.model)
    model = model_fn(num_classes=10, n_layers=full_depth, with_bn=True, dropout_p=0.0)
    
    trainloader, testloader = get_cifar10_special_pixel_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
        noise_level=args.noise_level,
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
            'model': args.model,
            'noise_level': args.noise_level,
            'seed': args.seed,
            'train_acc': metrics['final_train_acc'],
            'test_acc': metrics['final_test_acc'],
            'generalization_gap': metrics['generalization_gap'],
            'train_loss': metrics['final_train_loss'],
            'test_loss': metrics['final_test_loss'],
            'valid': True
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = f"{args.output_dir}/{args.model}_noise{args.noise_level:.3f}_seed{args.seed}_results.npz"
        np.savez(save_path, **result)
        print(f"Results saved to {save_path}")
        print(f"Final train acc: {metrics['final_train_acc']:.2f}%, test acc: {metrics['final_test_acc']:.2f}%")
    else:
        print(f"Warning: {args.model} achieved only {metrics['final_train_acc']:.2f}% train accuracy")
        result = {
            'model': args.model,
            'noise_level': args.noise_level,
            'seed': args.seed,
            'valid': False
        }
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = f"{args.output_dir}/{args.model}_noise{args.noise_level:.3f}_seed{args.seed}_results.npz"
        np.savez(save_path, **result)


if __name__ == "__main__":
    main()