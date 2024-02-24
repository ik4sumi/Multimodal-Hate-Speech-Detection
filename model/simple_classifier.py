import torch
import numpy as np
from torch import nn
from . import common

class SimpleClassifier(nn.Module):
    def __init__(self, in_channel=1024, out_channel=10, hid=128, layer_num=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channel, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 6),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
