import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
from PIL import Image

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
        image = img_to_array(image)
        image_tensor = transform_to_tensor(image)
        erased_tensor = random_erase(image_tensor)
        image = transform_to_image(erased_tensor)
        image.save(os.path.join(output_dir, filename))  # Save erased image

        # If further augmentation is needed
        image = image.reshape((1,) + image.shape)
        for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix='aug_', save_format='jpeg'):
            break  # Save one augmented image and break
