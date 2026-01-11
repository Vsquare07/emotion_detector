import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from model import myModel
from tqdm.auto import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 50
LR = 0.001

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

model = myModel(in_f=1, hid_f=64, out_f=7).to(device=DEVICE)
model.load_state_dict(torch.load("models/model4.pth"))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-4)
def acc_fn(pred,y):
    total = len(y)
    correct = 0
    for i in range(total):
        if(pred[i] == y[i]):
            correct += 1
    return (correct/total)*100

loss_arr = []
epoch_arr = []
for epoch in tqdm(range(EPOCHS)):
    net_loss = 0
    net_acc = 0
    for image, label in tqdm(train_dataloader):
        image, label = image.to(device=DEVICE), label.to(device=DEVICE)

        model.train()
        pred = model(image)
        optimizer.zero_grad()
        loss = loss_fn(pred, label)
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(pred)
        pred_labels = torch.argmax(pred_probs, dim=1)
        acc = acc_fn(pred_labels, label)
        net_loss += loss.item()
        net_acc += acc
        loss.backward()
        optimizer.step()

    net_loss /= len(train_dataloader)
    net_acc /= len(train_dataloader)
    loss_arr.append(net_loss)
    epoch_arr.append(epoch)
    print(f"Epoch : {epoch} ---> Loss : {net_loss:.3f}, Accuracy : {net_acc:.2f}%")
plt.plot(epoch_arr, loss_arr)
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.show()
torch.save(model.state_dict(), "models/model5.pth")