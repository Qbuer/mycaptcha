from importlib import import_module
from random import randint, random
import pytorch_lightning as pl
import os
from torch.utils import data
from torchvision import transforms as  T
import torch
from PIL import Image
from torch.utils.data import DataLoader
import re
import random

import any_captcha.config as C
from any_captcha.model.utils import CaptchaUtils
from any_captcha.model.capthafactory import CaptchaFactory
import copy
from captcha.image import ImageCaptcha
import string

ImageWidth = 100
ImageHeight = 40
charNumber = 4

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

alphabet = nums + lower_char + upper_char

def StrtoLabel(Str):
    label = []
    for i in range(0, charNumber):
        if Str[i] >= '0' and Str[i] <= '9':
            label.append(ord(Str[i]) - ord('0'))
        elif Str[i] >= 'a' and Str[i] <= 'z':
            label.append(ord(Str[i]) - ord('a') + 10)
        else:
            label.append(ord(Str[i]) - ord('A') + 36)
    return label

def LabeltoStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str

class Captcha(data.Dataset):
    def __init__(self, root, train=True):
        self.imgsPath = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_label(self, path : str):
        return path.split("/")[-1].split(".")[0]

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        label = self.get_label(imgPath) 
        assert len(label) == charNumber, label
        labelTensor = torch.Tensor(StrtoLabel(label)).long()
        data = Image.open(imgPath)
        data = self.transform(data.convert("RGB"))
        return (data, labelTensor, label)

    def __len__(self):
        return len(self.imgsPath)

class augCaptcha(Captcha):
    def __init__(self, root, train=True):
        Captcha.__init__(self, root, train)
        # super(augCaptcha, self).__init__(root, train)
        self.pattern = re.compile(r'\w+_original_(\d*\w*)')

    def get_label(self, path: str):
        return self.pattern.search(path).groups()[0]

class inferCaptcha(Captcha):
    def __init__(self, root, train=False):
        Captcha.__init__(self, root, train)

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        label = self.get_label(imgPath) 
        labelTensor = torch.Tensor(StrtoLabel('0000')).long()
        data = Image.open(imgPath)
        data = self.transform(data.convert("RGB"))
        return (data, labelTensor, label)

class infiniteCaptcha(data.Dataset):
    def __init__(self):

        color_num = len(C.colors)

        def custom_fn(single_char):
            return single_char

        def bg_custom_fn(bg):
            dots = random.randint(30, 60)
            for _ in range(dots):
                CaptchaUtils.draw_dot(bg, CaptchaUtils.get_rgb("0x7D7D7D"), width=1)

            for _ in range(5):
                if random.randint(0,2) == 0: continue 
                points = []
                for _ in range(2):
                    points.append(random.randint(0, 100))
                for _ in range(2):
                    points.append(random.randint(0, 40))

                width = random.randint(1,3)
                color = C.colors[random.randint(0, color_num)-1]
                CaptchaUtils.draw_line(bg, (points[0], points[2]), (points[1],points[3]),
                                   CaptchaUtils.get_rgb(color), width)
            
            return bg

        
        self.any_generator = []

        for font in C.fonts:
    
            config = copy.deepcopy(C.config)
            config['fonts'] = [font]
            colors_ = set()
            for _ in range(4):
                if random.randint(0,1): continue
                colors_.add(C.colors[random.randint(0, color_num-1)])
            if len(colors_) == 0:
                colors_.add(C.colors[random.randint(0, color_num-1)])
            config['colors'] = list(colors_)
            config['align'] = random.randint(1,2)
            self.any_generator.append(CaptchaFactory(char_custom_fns=[custom_fn], bg_custom_fns=[bg_custom_fn], **config))

        class helperImageCaptcha(ImageCaptcha):
            def __init__(self, *args, **kwargs):
                ImageCaptcha.__init__(self, *args, **kwargs)
            
            def generate_captcha(self):
                code = ''.join(random.sample(string.ascii_letters + string.digits, 4))
                image = self.generate_image(code)
                return image, code

        # self.any_generator.append(helperImageCaptcha(width=100, height=40))

        self.transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



    def __getitem__(self, index):
        idx = random.randint(1, len(self.any_generator)) - 1

        image, label = (self.any_generator[idx]).generate_captcha()
        data = self.transform(image)
        labelTensor = torch.Tensor(StrtoLabel(label)).long()
        return (data, labelTensor, label)

    def __len__(self):
        return 50000
            




class captchaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        self.captcha_train = Captcha(os.path.join(self.data_dir, 'train'), train=True)
        self.captcha_val = Captcha(os.path.join(self.data_dir, 'val'), train=True)
        self.aug_train = augCaptcha(os.path.join(self.data_dir, 'aug'), train=True)

        self.captcha_infer = inferCaptcha(os.path.join(self.data_dir, 'b_predict'), train=False)

        self.captcha_infinite = infiniteCaptcha()


    def train_dataloader(self):
        # return DataLoader(self.captcha_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return DataLoader(self.captcha_train+self.aug_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # return DataLoader(self.captcha_train+self.captcha_infinite, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.captcha_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.captcha_infer, batch_size=1, shuffle=False, num_workers=self.num_workers)



