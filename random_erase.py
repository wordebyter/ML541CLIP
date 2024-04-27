import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import save_img, img_to_array, ImageDataGenerator
from torchvision import transforms
from timm.data.random_erasing import RandomErasing
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import PIL
import torchvision.transforms as T

input_dir = 'crop50'
output_dir = 'erased'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image = load_img(f'/home/jyen/Desktop/deep_learning/final-proj/crop50/{filename}')
        x   = transforms.ToTensor()(image)
        random_erase = RandomErasing(probability=1, min_area = .02, max_area = 0.3, mode='pixel', device='cpu')
        transform = T.ToPILImage()
        img = transform(random_erase(x))
        prefix = os.path.splitext(filename)[0]
        img_path = os.path.join(input_dir, filename)
        prefix = os.path.splitext(filename)[0]
        # image = img_to_array(image)
        # image = image.reshape((1,) + image.shape)
       
        img.save(f"/home/jyen/Desktop/deep_learning/final-proj/erased/{prefix}.jpg")

    
