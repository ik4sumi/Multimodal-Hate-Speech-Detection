import torch
import numpy as np
from torch import nn
from . import common

class SimpleClassifier(nn.Module):
    def __init__(self, in_channel=512, out_channel=10, hid=128, layer_num=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channel, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_channel)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
