import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self, in_f, hid_f, out_f):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=hid_f, kernel_size=3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hid_f, out_channels=hid_f, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hid_f, out_channels=hid_f, kernel_size=3),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=hid_f, out_channels=hid_f, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hid_f*36*36, out_features=hid_f*36*36),
            nn.ReLU(),
            nn.Linear(in_features=hid_f*36*36, out_features=out_f)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x