import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import myModel

model = myModel(in_f=1, hid_f=10, out_f=7)
model.load_state_dict(torch.load("models/model1.pth"))

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
    with torch.inference_mode():
        img = img.unsqueeze(0)
        pred = model(img)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)
        pred_labels = torch.argmax(pred_probs, dim=1)
    color = "green" if pred_labels==label else "red"
    fig.add_subplot(r,c,i+1)
    plt.title(classes[label]+"->"+classes[pred_labels], color=color)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis(False)
plt.show()