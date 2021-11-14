import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplabv2 import Res_Deeplab
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d
from utils.helpers import colorize_mask

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p


def evaluate(model, ignore_label=250, save_dir=None):
    # palette
    NTHU_HUSKY_palette =  [
        128, 64, 128,
        244, 35, 232,
        70, 70, 70,
        107, 142, 35,
        152, 251, 152,
        0, 130, 180,
        220, 20, 60,
        0, 0, 142,
        119, 11, 32,
        255, 255, 0
    ]

    interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)

    # numpy array # test
    img_path = ''
    img = Image.open(img_path).convert('RGB')

    img = np.asarray(img, np.float32)
    img = img[:, :, ::-1]  # change to BGR
    img -= IMG_MEAN
    image = img.transpose((2, 0, 1))
    
    with torch.no_grad():
        output  = model(Variable(image).cuda())
        output = interp(output)
        output = output.cpu().data[0].numpy()
        output = np.asarray(np.argmax(output, axis=0), dtype=np.int)
        colorized_mask = colorize_mask(output, NTHU_HUSKY_palette)
        colorized_mask.save(os.path.join(save_dir, 'nthu_pred.png'))

def main():
    """Create the model and start the evaluation process."""


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = Res_Deeplab(num_classes=num_classes)

    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model'])
   
    model.cuda()
    model.eval()

    evaluate(model, ignore_label=ignore_label, save_dir=save_dir)


if __name__ == '__main__':
    model_path = 'testtttttt'

    config = torch.load(model_path)['config']
  
    num_classes = 9
        
    ignore_label = config['ignore_label']
    save_dir = os.path.join(*model_path.split('/')[:-1])
    save_dir = './results/nthu'
    
    
    main()
