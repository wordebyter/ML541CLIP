#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:07:03 2024

@author: jyen
"""
import json
from PIL import Image
import glob
import tensorflow as tf
#from tqdm import tqdm
import numpy as np
import fnmatch
#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader

#import clip
#from transformers import CLIPProcessor, CLIPModel
import os
from os import listdir
from skimage import io
#from sklearn.model_selection import train_test_split
#from transformers import CLIPConfig, CLIPModel


#path to json file and folder with images
json_path = os.path.expanduser('~/Desktop/deep_learning/final-proj/captions_val2017.json')
image_path = os.path.expanduser('~/Desktop/deep_learning/final-proj/val2017_copy')
filedir = os.path.expanduser('~/Desktop/deep_learning/final-proj/crop50')

input_data=[]
for images in os.listdir(image_path):
    obj = os.path.abspath(images)
    input_data.append(obj)
    

def random_crop(image, scale):
    width_orig = image.shape[0]
    height_orig = image.shape[1]
    NEW_IMG_WIDTH = width_orig*scale
    NEW_IMG_HEIGHT = height_orig*scale
    cropped_image = tf.image.random_crop(
      image, size=[NEW_IMG_HEIGHT, NEW_IMG_WIDTH, 3])
    cropped_image = np.resize(image, [width_orig, height_orig])
    with open(filedir, "wb") as fp:
            fp.write(cropped_image)
            
if __name__ == "__main__":
    print(input_data)
    for f in input_data:
        io.imshow(f)
        random_crop(f, .50)
        
