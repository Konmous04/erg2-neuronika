import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def show_images(indices, title, n=5):
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices[:n]):
        img = test_imgs[idx]
        label = y_test[idx]
        img = img*torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1)
        img = img+torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
        img = img.permute(1,2,0).numpy()

        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[label]
        pred_label = class_names[yt[idx]]
        plt.title(f"True: {true_label}\nPred: {pred_label}")

    plt.suptitle(title)
    plt.show()


def extract_classes(dataset, classes):
    X = []
    y = []
    imgs = [] 
    for img, label in dataset:
        if label in classes:
            img_flat = img.view(-1).numpy()
            imgs.append(img)
            X.append(img_flat)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, imgs

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

classes_to_keep = [3, 5]
X_train, y_train, _ = extract_classes(train_dataset, classes_to_keep)
X_test, y_test, test_imgs = extract_classes(test_dataset, classes_to_keep)

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

scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# 
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



C_values = [0.1, 1, 10]
gamma_values = [0.01, 0.1, 1]
results = []
for C in C_values:
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        
        start = time.perf_counter()
        svm.fit(X_train_pca, y_train)
        end = time.perf_counter()
        train_time = end-start

        y_train_pred = svm.predict(X_train_pca)
        y_test_pred = svm.predict(X_test_pca)
        if(C==1 and gamma==0.01):
            yt = y_test_pred

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        results.append((C, gamma, train_acc, test_acc, train_time))
        
        print(f"C={C}, gamma={gamma}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Training Time: {train_time:.2f} sec")

correct_idx = np.where(y_test==yt)[0]
wrong_idx = np.where(y_test!=yt)[0]
show_images(correct_idx, "Correctly Classified Images")
show_images(wrong_idx, "Misclassified Images")

