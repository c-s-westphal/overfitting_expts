import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split


def get_cifar10_transforms(augment=True):
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test


def get_cifar10_dataloaders(batch_size=128, num_workers=4, subset_size=None, seed=None):
    transform_train, transform_test = get_cifar10_transforms()
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    if subset_size is not None and subset_size < len(trainset):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        indices = np.arange(len(trainset))
        labels = np.array([trainset[i][1] for i in range(len(trainset))])
        
        _, subset_indices = train_test_split(
            indices, 
            train_size=subset_size,
            stratify=labels,
            random_state=seed
        )
        
        trainset = Subset(trainset, subset_indices[:subset_size])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


class SpecialPixelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, pixel_location=(16, 16), noise_level=0.0, seed=None):
        self.base_dataset = base_dataset
        self.pixel_location = pixel_location
        self.noise_level = noise_level
        
        if seed is not None:
            np.random.seed(seed)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        image = image.clone()
        
        label_value = label / 9.0
        
        noise = np.random.uniform(-self.noise_level, self.noise_level)
        pixel_value = label_value + noise
        pixel_value = np.clip(pixel_value, 0, 1)
        
        denorm_mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=image.dtype, device=image.device).view(3, 1, 1)
        denorm_std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=image.dtype, device=image.device).view(3, 1, 1)
        
        normalized_value = (pixel_value - denorm_mean) / denorm_std
        
        image[:, self.pixel_location[0], self.pixel_location[1]] = normalized_value.squeeze()
        
        return image, label


def get_cifar10_special_pixel_dataloaders(batch_size=128, num_workers=4, noise_level=0.0, seed=None,
                                          pixel_location=(16, 16), augment=True):
    transform_train, transform_test = get_cifar10_transforms(augment=augment)
    
    base_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    trainset = SpecialPixelDataset(base_trainset, pixel_location=pixel_location, noise_level=noise_level, seed=seed)
    
    base_testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Keep test set clean (no special pixel at test time)
    testset = base_testset
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader