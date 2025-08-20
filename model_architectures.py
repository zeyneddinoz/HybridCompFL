#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:42:04 2025

@author: zoz
"""


import torch.nn as nn
import torch



# Model definitions (from first code)
def get_model(model_name: str, input_channels: int = 1, num_classes: int = 10) -> nn.Module:
    class LeNet(nn.Module):
        def __init__(self, input_channels=input_channels):
            super(LeNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(16*4*4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    class AlexNet(nn.Module):
        def __init__(self, input_channels=1):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Linear(256*6*6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 10)
            )
    
        def forward(self, x):
            x = self.features(x)
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    class VGG16(nn.Module):
        def __init__(self, input_channels=input_channels):
            super(VGG16, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256*4*4, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256*4*4)
            x = self.classifier(x)
            return x

    models = {
        'LeNet': LeNet(input_channels=input_channels),
        'AlexNet': AlexNet(input_channels=input_channels),
        'VGG16': VGG16(input_channels=input_channels)
    }

    return models[model_name].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))