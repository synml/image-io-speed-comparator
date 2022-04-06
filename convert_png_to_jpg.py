import os
import glob
import torchvision

image_paths = glob.glob(os.path.join('data', '*.png'))

for image_path in image_paths:
    image = torchvision.io.read_image(image_path)
    torchvision.io.write_jpeg(image, image_path.replace('.png', '.jpg'), quality=100)
