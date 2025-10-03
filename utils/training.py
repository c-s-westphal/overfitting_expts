import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def train_model(model, trainloader, testloader, epochs=200, lr=0.1, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step()
        
        pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%'
        })
    
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_train_loss': train_losses[-1],
        'final_train_acc': final_train_acc,
        'final_test_loss': test_losses[-1],
        'final_test_acc': final_test_acc,
        'generalization_gap': final_train_acc - final_test_acc
    }