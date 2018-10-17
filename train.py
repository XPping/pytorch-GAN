# coding: utf-8

import argparse
from datasets.data_loader import get_loader
from gans import dcgan, wgan, began
from commons import commons


def train_dcgan(config):
    config.D_last_act = "sigmoid"
    config.G_last_act = "tanh"
    commons.make_dirs([config.model_save_path, config.log_save_path, config.image_save_path])
    data_loader = get_loader(celeba_path=config.celeba_image_path,
                             celeba_att_file=config.celeba_att_list_file,
                             image_size=config.image_size,
                             batch_size=config.batch_size)
    commons.set_gpu(1)
    solver = dcgan.Solver(data_loader, config)
    solver.train()
def train_wgan(config):
    config.D_last_act = ""
    config.G_last_act = "tanh"
    commons.make_dirs([config.model_save_path, config.log_save_path, config.image_save_path])
    data_loader = get_loader(celeba_path=config.celeba_image_path,
                             celeba_att_file=config.celeba_att_list_file,
                             image_size=config.image_size,
                             batch_size=config.batch_size)
    commons.set_gpu(1)
    solver = wgan.Solver(data_loader, config)
    solver.train()
def train_began(config):
    config.D_last_act = "tanh"
    config.G_last_act = "tanh"
    config.noise_n = 128
    commons.make_dirs([config.model_save_path, config.log_save_path, config.image_save_path])
    data_loader = get_loader(celeba_path=config.celeba_image_path,
                             celeba_att_file=config.celeba_att_list_file,
                             image_size=config.image_size,
                             batch_size=config.batch_size)
    commons.set_gpu(1)
    solver = began.Solver(data_loader, config)
    solver.train()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_n', type=int, default=100)
    parser.add_argument('--G_last_act', type=str, default="sigmoid")
    parser.add_argument('--D_out_n', type=int, default=1)
    parser.add_argument('--D_last_act', type=str, default="sigmoid")

    parser.add_argument('--G_lr', type=float, default=0.0001)
    parser.add_argument('--D_lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--lambda_k', type=float, default=0.001)

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--D_train_step', type=int, default=5)
    parser.add_argument('--save_image_step', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=10)

    parser.add_argument('--model_save_path', type=str, default="out/models")
    parser.add_argument('--log_save_path', type=str, default='out/log')
    parser.add_argument('--image_save_path', type=str, default='out/samples')

    parser.add_argument('--use_tensorboard', type=str, default=None)
    parser.add_argument('--pretrained_model', type=list, default=None)

    parser.add_argument('--celeba_image_path', type=str,
                        default="/home/xpp/data/celebA-face")
    parser.add_argument('--celeba_att_list_file', type=str,
                        default="/home/xpp/data/celebA-face.txt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)
    config = parser.parse_args()

    # train_dcgan(config)
    # train_wgan(config)
    train_began(config)


if __name__ == '__main__':
    main()

