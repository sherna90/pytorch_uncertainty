# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from torchvision.models.feature_extraction import create_feature_extractor

import os

class GaussHeadNet(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(GaussHeadNet,self).__init__()

        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, 4)

        self.fc1_var = nn.Linear(input_dim, hidden_dim)
        self.fc2_var = nn.Linear(hidden_dim, 4)

    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))

        mean = F.relu(self.fc1_mean(x_feature))  # (shape: (batch_size, hidden_dim))
        mean = self.fc2_mean(mean)  # (shape: batch_size, 1))

        log_var = F.relu(self.fc1_var(x_feature))  # (shape: (batch_size, hidden_dim))
        log_var = F.softplus(self.fc2_var(log_var))  # (shape: batch_size, 1))

        return mean, log_var


class GaussFeatureNet(nn.Module):
    def __init__(self,backbone):
        super(GaussFeatureNet,self).__init__()
        if backbone=='resnet34':
            self.weights=ResNet34_Weights.DEFAULT
            self.body=create_feature_extractor(
                resnet34(weights=self.weights), 
                return_nodes={'layer4':'last_layer'})
        elif backbone=='vgg16':
            self.weights=VGG16_Weights.DEFAULT
            self.body=create_feature_extractor(
                vgg16(weights=self.weights), 
                return_nodes={'features.30':'last_layer'})
        elif backbone=='mobilenet_v3':
            self.weights=MobileNet_V3_Large_Weights.DEFAULT
            self.body=create_feature_extractor(
                mobilenet_v3_large(weights=self.weights), 
                return_nodes={'features.16':'last_layer'})
        for param in self.body.parameters():
            param.requires_grad = False
        

    def forward(self, x):
        preprocess = self.weights.transforms()
        x=preprocess(x)
        x_feature = self.body(x) # (shape: (batch_size, 512))
        return x_feature['last_layer'].flatten(1)


class GaussNet(nn.Module):
    def __init__(self,backbone):
        super(GaussNet, self).__init__()
        self.feature_net = GaussFeatureNet(backbone)
        input=torch.rand(1,3,224,224)
        with torch.no_grad():
            output=self.feature_net(input)
        input_dim = output.shape[1:][0]
        hidden_dim=512
        self.head_net = GaussHeadNet(input_dim,hidden_dim)

    def forward(self, x):
        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.head_net(x_feature)



