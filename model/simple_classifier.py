import torch
import numpy as np
from torch import nn
from . import common

class SimpleClassifier(nn.Module):
    def __init__(self, in_channel=512, out_channel=10, hid=128, layer_num=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channel, 512),
            #nn.ReLU(inplace=True),
            #nn.Linear(512, 256),
            #nn.ReLU(inplace=True),
            nn.Linear(512, out_channel)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
