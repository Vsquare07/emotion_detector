import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from model import myModel
import numpy
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

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
model.load_state_dict(torch.load("models/model1.pth"))
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

y_preds = []
for image, label in tqdm(test_dataloader):
    with torch.inference_mode():
        pred = model(image)
        loss = loss_fn(pred, label)
        net_loss += loss.item()
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)
        pred_labels = torch.argmax(pred_probs, dim=1)
        y_preds.append(pred_labels)
        acc = acc_fn(pred_labels, label)
        net_acc += acc
net_acc /= len(test_dataloader)
net_loss /= len(test_dataloader)
print(net_acc, net_loss)

y_pred_tensor = torch.cat(y_preds)
confmat = ConfusionMatrix(num_classes=len(test_dataset.classes), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=torch.tensor(test_dataset.targets))

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=test_dataset.classes,
    figsize=(10,10)
)
plt.show()
