import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import myModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 1
LR = 0.01

train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True
)

classes = train_dataset.classes
fig = plt.figure(figsize=(10,10))
r, c = 4, 4
images, labels = next(iter(train_dataloader))
for i in range(16):
    img, label = images[i], labels[i]
    #TODO : add model predictions to label with red/green colour
    fig.add_subplot(4,4,i+1)
    plt.title(classes[label])
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis(False)
plt.show()