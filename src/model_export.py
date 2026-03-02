import torch
import torch.nn as nn

class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = BiggerNet()
dummy = torch.randn(1, 20)

torch.onnx.export(model, dummy, "model.onnx")
print("model.onnx exported successfully!")
