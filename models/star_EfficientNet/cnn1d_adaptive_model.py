#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Last Modified: 2024-11-20
# Modified By: H. Kang


import math
import torch
import torch.nn as nn

#value*scale
def round_value(value, scale):
    return int(math.ceil(value * scale))


# ================================================
'''
NAS - original setting of EfficientNet
layer : Conv1d -> BatchNorm1d -> ReLU

'''
class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(ConvBnRelu, self).__init__()
        pad_size = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=pad_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

class CNNmodule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, depth_scale, width_scale, initial=False):
        super(CNNmodule, self).__init__()
        
        depth = round_value(2, depth_scale) # depth scaling 
        width = round_value(out_channel, width_scale) # width scaling

        print(f"Creating CNN Module: Depth={depth}, Width={width}, Initial={initial}")

        if initial:
            layers = [ConvBnRelu(in_channel, width, kernel_size, stride=2)]
        else:
            layers = [ConvBnRelu(round_value(in_channel, width_scale), width, kernel_size, stride=2)] 

        for i in range(depth - 1):
            layers += [ConvBnRelu(width, width, kernel_size)]
        self.cnn_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn_module(x)

class CNN1d_adaptive(nn.Module):
    def __init__(self, kernel_size, num_classes, alpha, beta, phi):
        super(CNN1d_adaptive, self).__init__()

        depth_scale = alpha ** phi
        width_scale = beta ** phi
        self.last_channel = round_value(128, width_scale)

        self.feature = nn.Sequential(
            CNNmodule(1, 16, kernel_size, depth_scale, width_scale, initial=True),
            CNNmodule(16, 32, kernel_size, depth_scale, width_scale),
            CNNmodule(32, 64, kernel_size, depth_scale, width_scale),
            CNNmodule(64, 128, kernel_size, depth_scale, width_scale)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(-1, self.last_channel)
        out = self.fc(x)
        return out

# ================================================
'''
AdaptiveDF
Conv1d -> BatchNorm1d -> ELU -> MaxPool1d -> Dropuout
'''
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='relu'):
        super(ConvBnAct, self).__init__()
        padding = kernel_size // 2  # 수정된 패딩 값
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.ELU(alpha=1.0)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))



class AdaptiveDFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, depth_scale, width_scale, initial=False):
        super(AdaptiveDFBlock, self).__init__()
        self.layers = nn.ModuleList()

        # Adjust width and depth using scaling
        depth = round_value(2, depth_scale)

        # Initial layer
        self.layers.append(ConvBnAct(in_channels, out_channels, kernel_size, stride=1, activation='elu'))

        # Add depth layers
        for _ in range(depth - 1):
            self.layers.append(ConvBnAct(out_channels, out_channels, kernel_size, stride=1, activation='elu'))

        # Pooling and Dropout
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size, padding=pool_size // 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            print(f"AdaptiveDFBlock Layer {i}: Input shape {x.shape}")
            x = layer(x)
            print(f"AdaptiveDFBlock Layer {i}: Output shape {x.shape}")
        x = self.pool(x)
        x = self.dropout(x)
        return x





class AdaptiveDFNet(nn.Module):
    def __init__(self, input_shape, num_classes, alpha, beta, phi, kernel_size=8):
        super(AdaptiveDFNet, self).__init__()

        depth_scale = alpha ** phi
        width_scale = beta ** phi

        # 명시적으로 채널 연결
        self.feature_extractor = nn.Sequential(
            AdaptiveDFBlock(1, 32, kernel_size, pool_size=8, depth_scale=depth_scale, width_scale=width_scale, initial=True),
            AdaptiveDFBlock(32, 64, kernel_size, pool_size=8, depth_scale=depth_scale, width_scale=width_scale),
            AdaptiveDFBlock(64, 128, kernel_size, pool_size=8, depth_scale=depth_scale, width_scale=width_scale),
            AdaptiveDFBlock(128, 256, kernel_size, pool_size=8, depth_scale=depth_scale, width_scale=width_scale)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        for i, module in enumerate(self.feature_extractor):
            print(f"Block {i}: Input shape {x.shape}")
            x = module(x)
            print(f"Block {i}: Output shape {x.shape}")
        x = self.global_pool(x)
        print(f"After Global Pooling: {x.shape}")
        x = self.classifier(x)
        print(f"After Classifier: {x.shape}")
        return x





# ================================================

if __name__ == "__main__":
    #### test block 
    x = torch.rand(16, 1, 5000)
    model = CNN1d_adaptive(3, 5, 1, 1, 1)
    out = model(x)
