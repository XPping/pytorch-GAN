# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from torchvision.utils import save_image

import numpy as np
import os

from models.generator import Generator
from models.discriminator import BEGAN_Discriminator
from commons.commons import noise_vector

def last_act(act):
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    else:
        return None

class Solver(object):
    def __init__(self, data_loader, config):

        self.data_loader = data_loader

        self.noise_n = config.noise_n
        self.G_last_act = last_act(config.G_last_act)
        self.D_out_n = config.D_out_n
        self.D_last_act = last_act(config.D_last_act)

        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.D_train_step = config.D_train_step
        self.save_image_step = config.save_image_step
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.model_save_path = config.model_save_path
        self.log_save_path = config.log_save_path
        self.image_save_path = config.image_save_path

        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.build_model()

        if self.use_tensorboard is not None:
            self.build_tensorboard()
        if self.pretrained_model is not None:
            if len(self.pretrained_model) != 2:
                raise "must have both G and D pretrained parameters, and G is first, D is second"
            self.load_pretrained_model()

    def build_model(self):
        self.G = Generator(self.noise_n, self.G_last_act)
        self.D = BEGAN_Discriminator(self.noise_n//2, self.D_last_act)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.G_lr, [self.beta1, self.beta2])
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), self.D_lr, [self.beta1, self.beta2])
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def build_tensorboard(self):
        from commons.logger import Logger
        self.logger = Logger(self.log_save_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(self.pretrained_model[0]))
        self.D.load_state_dict(torch.load(self.pretrained_model[1]))

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        print(len(self.data_loader))
        k = 0.0
        for e in range(self.epoch):
            for i, batch_images in enumerate(self.data_loader):
                batch_size = batch_images.size(0)
                real_x = self.to_var(batch_images)
                noise_x = self.to_var(torch.FloatTensor(noise_vector(batch_size, self.noise_n)))
                # train D
                fake_x = self.G(noise_x)
                real_out = self.D(real_x)
                fake_out = self.D(fake_x.detach())
                # print(real_out.shape, real_x.shape, fake_x.shape, fake_out.shape)
                D_real = torch.mean(torch.abs(real_out - real_x))
                D_fake = torch.mean(torch.abs(fake_out - fake_x))
                D_loss = D_real - k * D_fake

                self.reset_grad()
                D_loss.backward()
                self.D_optimizer.step()
                # Log
                loss = {}
                loss['D/loss_real'] = D_real.data[0]
                loss['D/loss_fake'] = D_fake.data[0]
                loss['D/loss'] = D_loss.data[0]
                # Train G
                if (i + 1) % self.D_train_step == 0:
                    fake_x = self.G(noise_x)
                    fake_out = self.D(fake_x)
                    G_loss = torch.mean(torch.abs(fake_out - fake_x))
                    self.reset_grad()
                    G_loss.backward()
                    self.G_optimizer.step()
                    loss['G/loss'] = G_loss.data[0]
                # Update K
                balance = (self.gamma * D_real - D_fake).data[0]
                k += self.lambda_k * balance
                k = max(min(1, k), 0)
                loss['K'] = k
                loss['M'] = D_real.cpu().data[0] + np.abs(balance.cpu())
                # Print log
                if (i+1) % self.log_step == 0:
                    log = "Epoch: {}/{}, Iter: {}/{}".format(e+1, self.epoch, i+1, len(self.data_loader))
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, e * len(self.data_loader) + i+1)
            # Save images
            if (e+1) % self.save_image_step == 0:
                noise_x = self.to_var(torch.FloatTensor(noise_vector(32, self.noise_n)))
                fake_image = self.G(noise_x)
                save_image(self.denorm(fake_image.data),
                           os.path.join(self.image_save_path, "{}_fake.png".format(e + 1)))
            if (e+1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(), os.path.join(self.model_save_path, "{}_G.pth".format(e+1)))
                torch.save(self.D.state_dict(), os.path.join(self.model_save_path, "{}_D.pth".format(e+1)))