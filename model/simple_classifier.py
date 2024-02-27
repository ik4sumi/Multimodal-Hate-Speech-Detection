import torch
import numpy as np
from torch import nn
from . import common

class SimpleClassifier(nn.Module):
    def __init__(self, in_channel=512, out_channel=10, hid=128, layer_num=5,image_only=False,text_only=False):
        super().__init__()
        if image_only or text_only:
            self.in_channel = 512
        else:
            self.in_channel = 1024
        self.classifier = nn.Sequential(
            nn.Linear(self.in_channel, 2048),
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
        self.image_only = image_only
        self.text_only = text_only


    def forward(self, batch):
        x,y=batch["image"],batch["text"]
        if self.image_only and not self.text_only:
            x = torch.squeeze(x)
            output = self.classifier(x)
        elif self.text_only and not self.image_only:
            y = torch.squeeze(y)
            output = self.classifier(y)
        else:
            conbined = torch.cat((x,y),dim=-1)
            conbined = torch.squeeze(conbined)
            output = self.classifier(conbined)
        return output
