import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from model import myModel
from tqdm.auto import tqdm

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

model = myModel(in_f=1, hid_f=10, out_f=7)
model.load_state_dict(torch.load("models/model0.pth"))
loss_fn = nn.CrossEntropyLoss()
def acc_fn(pred,y):
    total = len(y)
    correct = 0
    for i in range(total):
        if(pred[i] == y[i]):
            correct += 1
    return (correct/total)*100

net_loss = 0
net_acc = 0
for image, label in tqdm(test_dataloader):
    with torch.inference_mode():
        pred = model(image)
        loss = loss_fn(pred, label)
        net_loss += loss.item()
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)
        pred_labels = torch.argmax(pred_probs, dim=1)
        acc = acc_fn(pred_labels, label)
        net_acc += acc
net_acc /= len(test_dataloader)
net_loss /= len(test_dataloader)
print(net_acc, net_loss)

#TODO: add confusion matrix
