import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing.image import save_img

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

for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        img_path = os.path.join(input_dir, filename)
        image = load_img(img_path)
        image = img_to_array(image)
        image = image.reshape((1,) + image.shape)
        
        prefix = os.path.splitext(filename)[0]

        i = 0
        for batch in datagen.flow(image, batch_size=1, save_to_dir=output_dir, save_prefix=prefix, save_format='jpeg'):
            i += 1
            if i > 20:
                break

