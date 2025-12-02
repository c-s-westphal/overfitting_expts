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


def get_mnist_transforms(augment=False):
    """
    Get MNIST transforms.

    Note: MNIST typically doesn't use augmentation, but we provide the option.
    """
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return transform_train, transform_test


def get_mnist_dataloaders(batch_size=128, num_workers=4, augment=False):
    """Get MNIST train and test dataloaders."""
    transform_train, transform_test = get_mnist_transforms(augment=augment)

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


class SpecialPixelDatasetMNIST(torch.utils.data.Dataset):
    """
    MNIST dataset with a special pixel that encodes the label with noise.

    Similar to SpecialPixelDataset but for MNIST (1 channel instead of 3).
    """
    def __init__(self, base_dataset, pixel_location=(14, 14), noise_level=0.0, seed=None):
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

        # Normalize label to [0, 1] range
        label_value = label / 9.0

        # Add noise
        noise = np.random.uniform(-self.noise_level, self.noise_level)
        pixel_value = label_value + noise
        pixel_value = np.clip(pixel_value, 0, 1)

        # Convert to normalized value (MNIST mean=0.1307, std=0.3081)
        denorm_mean = torch.tensor([0.1307], dtype=image.dtype, device=image.device).view(1, 1, 1)
        denorm_std = torch.tensor([0.3081], dtype=image.dtype, device=image.device).view(1, 1, 1)

        normalized_value = (pixel_value - denorm_mean) / denorm_std

        # Set the special pixel (MNIST has 1 channel)
        image[:, self.pixel_location[0], self.pixel_location[1]] = normalized_value.squeeze()

        return image, label


def get_mnist_special_pixel_dataloaders(batch_size=128, num_workers=4, noise_level=0.0, seed=None,
                                        pixel_location=(14, 14), augment=False):
    """
    Get MNIST dataloaders with special pixel encoding label information.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        noise_level: Noise level for special pixel (0.0 = perfect MI)
        seed: Random seed for reproducibility
        pixel_location: (row, col) for special pixel (default: center at 14, 14)
        augment: Whether to use data augmentation (typically False for MNIST)
    """
    transform_train, transform_test = get_mnist_transforms(augment=augment)

    base_trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )

    trainset = SpecialPixelDatasetMNIST(base_trainset, pixel_location=pixel_location,
                                        noise_level=noise_level, seed=seed)

    base_testset = torchvision.datasets.MNIST(
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


def get_mnist_binary_dataloaders(batch_size=128, num_workers=4, augment=False):
    """
    Get MNIST dataloaders for binary classification (digits 0 vs 1 only).

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment: Whether to use data augmentation

    Returns:
        trainloader, testloader (label 0 = digit 0, label 1 = digit 1)
    """
    transform_train, transform_test = get_mnist_transforms(augment=augment)

    # Load full MNIST
    full_trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    full_testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Filter to only digits 0 and 1
    train_indices = [i for i, (_, label) in enumerate(full_trainset) if label in [0, 1]]
    test_indices = [i for i, (_, label) in enumerate(full_testset) if label in [0, 1]]

    trainset = Subset(full_trainset, train_indices)
    testset = Subset(full_testset, test_indices)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def get_celeba_transforms(augment=True, image_size=64):
    """
    Get CelebA transforms with center crop and resize.

    Args:
        augment: Whether to use data augmentation (horizontal flip)
        image_size: Target image size (default: 64)

    Returns:
        transform_train, transform_test
    """
    # Standard ImageNet normalization (commonly used for CelebA)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        transform_train = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return transform_train, transform_test


def get_celeba_dataloaders(batch_size=128, num_workers=4, augment=True,
                           image_size=64, seed=None, data_root='./data'):
    """
    Get CelebA dataloaders for Male/Female classification.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment: Whether to use data augmentation (horizontal flip)
        image_size: Target image size (default: 64)
        seed: Random seed for reproducibility
        data_root: Root directory for data

    Returns:
        trainloader, testloader (Male=1, Female=0)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    transform_train, transform_test = get_celeba_transforms(augment=augment, image_size=image_size)

    # Monkey-patch CelebA to skip MD5 integrity check (we downloaded from Kaggle, not Google Drive)
    original_check = torchvision.datasets.CelebA._check_integrity
    def skip_integrity_check(self):
        # Just check if img_align_celeba directory exists
        import os
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))
    torchvision.datasets.CelebA._check_integrity = skip_integrity_check

    # CelebA Male attribute is at index 20 (0-indexed)
    # We'll use target_type='attr' and then extract the Male column
    # download=False because we download manually from Kaggle to avoid Google Drive quota
    trainset = torchvision.datasets.CelebA(
        root=data_root,
        split='train',
        target_type='attr',
        download=False,
        transform=transform_train
    )

    testset = torchvision.datasets.CelebA(
        root=data_root,
        split='valid',  # Use validation split as test set
        target_type='attr',
        download=False,
        transform=transform_test
    )

    # Restore original integrity check
    torchvision.datasets.CelebA._check_integrity = original_check

    # Wrap datasets to extract only Male attribute (index 20)
    class MaleClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, celeba_dataset, male_attr_idx=20):
            self.celeba_dataset = celeba_dataset
            self.male_attr_idx = male_attr_idx

        def __len__(self):
            return len(self.celeba_dataset)

        def __getitem__(self, idx):
            img, attrs = self.celeba_dataset[idx]
            # Extract Male attribute (0 or 1)
            label = attrs[self.male_attr_idx].item()
            return img, label

    trainset = MaleClassificationDataset(trainset)
    testset = MaleClassificationDataset(testset)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader