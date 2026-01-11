import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from model import myModel

model = myModel(in_f=1, hid_f=64, out_f=7)
model.load_state_dict(torch.load("models/model4(best).pth"))

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=True
)

classes = test_dataset.classes
fig = plt.figure(figsize=(10,10))
r, c = 4, 4
images, labels = next(iter(test_dataloader))
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