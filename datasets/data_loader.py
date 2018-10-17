# coding: utf-8

import torch
import os
import codecs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, celeba_path, celeba_att_file, transform):
        super(MyDataset, self).__init__()

        self.celeba_path = celeba_path
        self.celeba_att_file = celeba_att_file
        self.transform = transform

        print("Start load image..")
        self.preprocess()
        print("End load image")

    def preprocess(self):
        self.filenames = []
        with open(self.celeba_att_file, 'r') as fp:
            lines = fp.readlines()
            # self.images_num = int(lines[0])
            # celeba_att = lines[1].strip().split()
            for i in range(2, len(lines)):
                image_id = lines[i].strip().split()[0]
                self.filenames.append(image_id)
    def __getitem__(self, item):
        image = Image.open(os.path.join(self.celeba_path, self.filenames[item]))

        return self.transform(image)
    def __len__(self):
        return len(self.filenames)

def get_loader(celeba_path, celeba_att_file, image_size, batch_size=16):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MyDataset(celeba_path=celeba_path, celeba_att_file=celeba_att_file, transform=transform)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return data_loader