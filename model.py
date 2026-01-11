import torch
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self, in_f, hid_f, out_f):
        super().__init__()

        # Block 1: 48x48 -> 24x24
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_f, hid_f, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_f),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 24x24 -> 12x12
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid_f, hid_f*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_f*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 12x12 -> 6x6
        self.layer3 = nn.Sequential(
            nn.Conv2d(hid_f*2, hid_f*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_f*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hid_f*4*6*6, hid_f*2),
            nn.ReLU(),
            nn.Linear(hid_f*2, out_f)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    model = myModel(in_f=1, hid_f=64, out_f=7)
    batch_size = 16
    summary(model, input_size=(batch_size, 1, 48, 48))