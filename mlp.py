import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt

class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.data = []
        self.targets = []

        for img, label in cifar_dataset:
            if label==3:
                self.data.append(img)
                self.targets.append(0)
            elif label==5:
                self.data.append(img)
                self.targets.append(1)
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class simpleMLP(nn.Module):
    def __init__(self, h1=512):
        super(simpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, h1)
        self.relu1 = nn.ReLU()
    
        self.fc2 = nn.Linear(h1, 2)
        

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss/total
    epoch_accuracy = correct/total
    return epoch_loss, epoch_accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():    
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()*images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
    ])

    train_cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dataset = CatDogDataset(train_cifar)
    test_dataset = CatDogDataset(test_cifar)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = simpleMLP(h1=512).to(device)

    criterion = nn.MultiMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    s = time.perf_counter()
    for epoch in range(num_epochs):
        
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        end = time.perf_counter()
        total_time = end-start

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | "
              f"Time: {total_time:.1f}sec"
             )
    e = time.perf_counter()
    tt = e-s
    print(f"Total time for {num_epochs} epochs: {tt:.1f}sec")