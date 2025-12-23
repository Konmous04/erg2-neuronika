import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
import time

def loader_to_numpy(dataloader):
    X_list, y_list = [], []
    for images, labels in dataloader:
        X_list.append(images.view(images.size(0), -1).numpy())
        y_list.append(labels.numpy())
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X,y

def filter_cat_dog(X, y):
    mask = (y==3) | (y==5)
    X = X[mask]
    y = y[mask]
    y = np.where(y==3,0,1)
    return X, y

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform 
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


X_train, y_train = loader_to_numpy(trainloader)
X_test, y_test = loader_to_numpy(testLoader)

X_train, y_train = filter_cat_dog(X_train, y_train)
X_test, y_test = filter_cat_dog(X_test, y_test)


start = time.perf_counter()
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
y_pred_knn1 = knn1.predict(X_test)
acc_knn1 = accuracy_score(y_test, y_pred_knn1)*100
end = time.perf_counter()
knn1_time = end-start

start = time.perf_counter()
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
y_pred_knn3 = knn3.predict(X_test)
acc_knn3 = accuracy_score(y_test, y_pred_knn3)*100
end = time.perf_counter()
knn3_time = end-start

start = time.perf_counter()
ncc = NearestCentroid()
ncc.fit(X_train, y_train)
y_pred_ncc = ncc.predict(X_test)
acc_ncc = accuracy_score(y_test, y_pred_ncc)*100
end = time.perf_counter()
ncc_time = end-start


print(f"Ακρίβεια 1-NN: {acc_knn1:.2f}% σε {knn1_time:.2f}sec")
print(f"Ακρίβεια 3-NN: {acc_knn3:.2f}% σε {knn3_time:.2f}sec")
print(f"Ακρίβεια Nearest Centroid: {acc_ncc:.2f}% σε {ncc_time:.2f}sec")