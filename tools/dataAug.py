import Augmentor
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
# from parameters import *
import torch as t

import re
# from dataset import *

def get_distortion_pipline(path, num):
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p

def get_skew_tilt_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5,magnitude=0.02)
    p.skew_left_right(probability=0.5,magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.skew_corner(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p

def get_rotate_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.rotate(probability=1,max_left_rotation=1,max_right_rotation=1)
    p.sample(num)
    return p

if __name__ == "__main__":
    times = 2
    path = "/home/zhqiao/workspace/mycaptcha/data/dataset4/train/"
    
    num = len(os.listdir(path)) * times
    p = get_distortion_pipline(path, num)
    p.process()

    # p = Augmentor.Pipeline(path)
    # p.skew_tilt(probability=0.5, magnitude=0.02)
    # p.sample(num)
    # p.porcess()

    # p = Augmentor.Pipeline(path)
    # p.shear(probability=1,max_shear_left=25,max_shear_right=25)
    # p.sample(num)
    # p.process()
