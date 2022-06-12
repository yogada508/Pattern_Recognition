import os
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
# neural network modules
import torch.nn as nn
# optimizers
import torch.optim as optim
# transformations 
import torchvision.transforms as transforms

from torchvision import models
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

# calculate the mean and std of the cifar10 dataset
def get_mean_and_std(x_train):
    mean = []
    std = []
    for i in range(3):
        mean.append(np.mean((x_train/255)[:, :, :, i]))
        std.append(np.std((x_train/255)[:, :, :, i]))

    mean = np.array(mean)
    std = np.array(std)

    return mean, std

class Cifar10(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.data)

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
y_train = y_train.squeeze()

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
y_test = y_test.squeeze()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 64

# data augmentation with resize and normalization (project data to [-1, 1])
mean, std = get_mean_and_std(x_train)
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),        
        transforms.Normalize(mean, std)
    ])

test_dataset = Cifar10(x_test, y_test, transform=transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def model_predict(model, test_dataloader):
    predictions = np.array([])

    with torch.no_grad():
        model.eval()

        for images, _ in test_dataloader:
            images = images.to(device)

            outputs = model(images)

            # value, index
            v, pred = torch.max(outputs, 1)

            pred = pred.cpu().numpy()

            predictions = np.concatenate((predictions, pred), axis=None)

    return predictions



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
model = models.densenet121()

# reset final fully connected layer (num_ftrs = 1024)
num_ftrs = model.classifier.in_features

model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.2),
                        nn.Linear(256, 10))

# copy weights from weights file
model.load_state_dict(torch.load('HW5_weights.pth'))

# move model to a device
model = model.to(device)

# predict
print("Start to predict result")
y_pred = model_predict(model, test_dataloader)
y_test = np.load("y_test.npy")
print("Accuracy of my model on test set: ", accuracy_score(y_test, y_pred))