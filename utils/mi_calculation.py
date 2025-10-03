import numpy as np
from scipy.stats import entropy


def calculate_mutual_information(noise_level):
    if noise_level >= 0.5:
        return 0.0
    
    num_classes = 10
    
    max_mi = np.log2(num_classes)
    
    mi = max_mi * (1 - 2 * noise_level)
    
    return max(0, mi)


def get_noise_levels_with_mi(num_levels=10):
    noise_levels = np.linspace(0, 0.5, num_levels)
    
    mi_values = [calculate_mutual_information(noise) for noise in noise_levels]
    
    return noise_levels, mi_values


def theoretical_mi_uniform_noise(noise_level, num_classes=10):
    if noise_level == 0:
        return np.log2(num_classes)
    
    label_range = 1.0 / num_classes
    
    overlap_prob = min(1.0, 2 * noise_level / label_range)
    
    if overlap_prob >= 1.0:
        return 0.0
    
    mi = np.log2(num_classes) * (1 - overlap_prob)
    
    return max(0, mi)