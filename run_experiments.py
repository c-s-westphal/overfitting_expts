#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess


def run_experiment_1():
    print("Running Experiment 1: Varying Dataset Sizes")
    print("=" * 50)
    subprocess.run([sys.executable, "experiments/exp1_dataset_size.py"])


def run_experiment_2():
    print("Running Experiment 2: Special Pixel with Noise")
    print("=" * 50)
    subprocess.run([sys.executable, "experiments/exp2_special_pixel.py"])


def run_experiment_3():
    print("Running Experiment 3: Depth sweep at fixed MI=2.5 bits")
    print("=" * 50)
    subprocess.run([sys.executable, "experiments/exp3_depth_sweep.py"])


def print_results():
    from utils.results_utils import print_summary
    
    models = ['ResNet20', 'ResNet32', 'ResNet56', 'VGG11', 'VGG16', 'VGG19']
    
    print("\nExperiment 1 Results Summary:")
    print("=" * 60)
    for model in models:
        try:
            print_summary('exp1', model)
        except FileNotFoundError:
            print(f"{model} - Experiment exp1: No results found")
    
    print("\nExperiment 2 Results Summary:")
    print("=" * 60)
    for model in models:
        try:
            print_summary('exp2', model)
        except FileNotFoundError:
            print(f"{model} - Experiment exp2: No results found")


def main():
    parser = argparse.ArgumentParser(description='Run overfitting experiments')
    parser.add_argument('--exp1', action='store_true', help='Run experiment 1 (varying dataset sizes)')
    parser.add_argument('--exp2', action='store_true', help='Run experiment 2 (special pixel with noise)')
    parser.add_argument('--exp3', action='store_true', help='Run experiment 3 (depth sweep at fixed MI)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--results', action='store_true', help='Print results summary')
    parser.add_argument('--plot', action='store_true', help='Generate plots from results')
    
    args = parser.parse_args()
    
    if args.all or args.exp1:
        run_experiment_1()
    
    if args.all or args.exp2:
        run_experiment_2()

    if args.all or args.exp3:
        run_experiment_3()
    
    if args.results:
        print_results()
    
    if args.plot:
        print("Generating plots...")
        subprocess.run([sys.executable, "plot_results.py"])
    
    if not any([args.exp1, args.exp2, args.all, args.results, args.plot]):
        parser.print_help()


if __name__ == "__main__":
    main()