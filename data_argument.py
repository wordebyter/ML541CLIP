import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
from PIL import Image
import numpy as np

input_dir = '/home/zyang12/hw/MLfinal/ML541CLIP/val2017'
output_dir = '/home/zyang12/hw/MLfinal/ML541CLIP/images_augmented'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

random_erase = RandomErasing(probability=1, min_area=.02, max_area=0.3, mode='pixel', device='cpu')
transform_to_tensor = transforms.ToTensor()
transform_to_image = transforms.ToPILImage()

for filename in os.listdir(input_dir):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        image = load_img(image_path)
        image_array = img_to_array(image)
        
        image_array = np.expand_dims(image_array, axis=0)

        image_tensor = transform_to_tensor(image)
        erased_tensor = random_erase(image_tensor.unsqueeze(0))
        erased_image = transform_to_image(erased_tensor.squeeze(0))

        erased_image_array = np.expand_dims(img_to_array(erased_image), axis=0)

        prefix = os.path.splitext(filename)[0]
        for i in range(1):
            save_prefix = f"{prefix}_aug_{i+1}"
            for batch in datagen.flow(erased_image_array, batch_size=1, save_to_dir=output_dir, save_prefix=save_prefix, save_format='jpeg'):
                break  
    
