# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Generator(nn.Module):
    def __init__(self, noise_in, last_act):
        super(Generator, self).__init__()
        self.noise_in = noise_in
        self.last_act = last_act

        self.fc1 = nn.Linear(self.noise_in, 4 * 4 * 1024)
        # self.fc1_norm = nn.BatchNorm2d(1024 * 4 * 4, affine=True)
        self.fc1_relu = nn.LeakyReLU(0.02, inplace=True)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # 4 -> 8
        # self.norm1 = nn.BatchNorm2d(512, affine=True)
        self.de_relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)   # 8 -> 16
        # self.norm2 = nn.BatchNorm2d(256, affine=True)
        self.de_relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   # 16 -> 32
        # self.norm3 = nn.BatchNorm2d(128, affine=True)
        self.de_relu3 = nn.LeakyReLU(0.02, inplace=True)

        self.deconv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)     # 32 -> 64
        self.de_relu4 = last_act

    def forward(self, x):
        h = self.fc1_relu(self.fc1(x))
        h = h.view(-1, 1024, 4, 4)
        h = self.de_relu1(self.deconv1(h))
        h = self.de_relu2(self.deconv2(h))
        h = self.de_relu3(self.deconv3(h))
        h = self.deconv4(h) if self.de_relu4 is None else self.de_relu4(self.deconv4(h))

        return h