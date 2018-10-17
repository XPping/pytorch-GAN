# coding: utf-8

import numpy as np
import os

# generator noise vector
def noise_vector(batch_size, length):
    return np.random.normal(0., 1., size=[batch_size, length])

# make dirs for model output
def make_dirs(names):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)

def set_gpu(gpu_id):
    print("Using GPU id = {}".format(gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)