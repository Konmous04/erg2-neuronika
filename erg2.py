import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

def extract_classes(dataset, classes):
    X = []
    y = []
    for img, label in dataset:
        if label in classes:
            img_flat = img.view(-1).numpy()
            X.append(img_flat)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

classes_to_keep = [3, 5]
X_train, y_train = extract_classes(train_dataset, classes_to_keep)
X_test, y_test = extract_classes(test_dataset, classes_to_keep)

pca_temp = PCA()
pca_temp.fit(X_train)
explained_variance = np.cumsum(pca_temp.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.90) + 1
print("Αριθμός συνιστωσών για 90% πληροφορίας:", n_components)

pca = PCA(n_components=n_components)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Train PCA shape:", X_train_pca.shape)
print("Test PCA shape:", X_test_pca.shape)

svm_linear = SVC(kernel='linear')

start = time.perf_counter()
svm_linear.fit(X_train_pca, y_train)
end = time.perf_counter()
train_time = end-start

y_train_pred = svm_linear.predict(X_train_pca)
y_test_pred = svm_linear.predict(X_test_pca)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Linear SVM")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time:.2f} sec")



svm_rdf = SVC(kernel='rbf')
start = time.perf_counter()
svm_rdf.fit(X_train_pca, y_train)
end = time.perf_counter()
train_time = end-start

y_train_pred = svm_rdf.predict(X_train_pca)
y_test_pred = svm_rdf.predict(X_test_pca)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("RΒF SVM")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time:.2f} sec")