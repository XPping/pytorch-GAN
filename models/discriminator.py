# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.generator import Generator

class Discriminator(nn.Module):
    def __init__(self, n_out, last_act):
        super(Discriminator, self).__init__()
        self.n_out = n_out
        self.last_act = last_act

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        # self.norm1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # 32 -> 16
        # self.norm2 = nn.BatchNorm2d(128, affine=True)
        self.relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)   # 16 -> 8
        # self.norm3 = nn.BatchNorm2d(256, affine=True)
        self.relu3 = nn.LeakyReLU(0.02, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)     # 8 -> 4
        # self.norm4 = nn.BatchNorm2d(512, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(4 * 4 * 512, self.n_out)
        self.fc1_relu = last_act

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        # print("D", h.shape)
        h = self.relu2(self.conv2(h))
        # print("D", h.shape)
        h = self.relu3(self.conv3(h))
        # print("D", h.shape)
        h = self.relu4(self.conv4(h))
        # print("D", h.shape)
        h = h.view(-1, 4 * 4 * 512)
        h = self.fc1(h) if self.fc1_relu is None else self.fc1_relu(self.fc1(h))

        return h


class BEGAN_Discriminator(nn.Module):
    def __init__(self, encode_out, last_act):
        super(BEGAN_Discriminator, self).__init__()

        self.enc = Discriminator(encode_out, nn.LeakyReLU(0.02, inplace=True))
        self.dec = Generator(encode_out, last_act)

    def forward(self, x):
        h = self.enc(x)
        h = self.dec(h)
        return h