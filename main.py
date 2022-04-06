import os
import glob
import time

import cv2
import jpeg4py
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image


image_paths = glob.glob(os.path.join('data', '*.jpg'))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

for _ in range(5):
    cv_time = time.time()
    for image_path in image_paths:
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    cv_time = time.time() - cv_time

    jpeg4py_time = time.time()
    for image_path in image_paths:
        image = jpeg4py.JPEG(image_path).decode()
    jpeg4py_time = time.time() - jpeg4py_time

    pil_time = time.time()
    for image_path in image_paths:
        image = TF.pil_to_tensor(Image.open(image_path).convert('RGB'))
    pil_time = time.time() - pil_time

    torchvision_time = time.time()
    for image_path in image_paths:
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)
    torchvision_time = time.time() - torchvision_time

    print(f'cv_time: \t\t{cv_time:.2f}')
    print(f'jpeg4py_time: \t\t{jpeg4py_time:.2f}')
    print(f'pil_time: \t\t{pil_time:.2f}')
    print(f'torchvision_time: \t{torchvision_time:.2f}')
    print('-------------------------------')
