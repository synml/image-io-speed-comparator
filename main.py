import argparse
import os
import gc
import glob
import time

import cv2
import jpeg4py
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

import utils


"""
로드한 결과는 OpenCV와 jpeg4py가 같고, PIL과 Torchvision.io가 같음.
그러나 OpenCV, jpeg4py와 PIL, Torchvision.io는 서로 다름.
(OpenCV == jpeg4py) != (PIL == Torchvision.io)

픽셀 값에서 1 ~ 3정도 차이 발생
"""

parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=5, help='number of iterations')
args = parser.parse_args()

image_paths = glob.glob(os.path.join('data', '*.jpg'))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

total_cv_time = []
total_jpeg4py_time = []
total_pil_time = []
total_torchvision_time = []

for i in range(args.repeat):
    # Calculate OpenCV
    cv_time = time.time()
    for image_path in image_paths:
        image = torch.as_tensor(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).permute((2, 0, 1))
    cv_time = time.time() - cv_time
    del image
    gc.collect()

    # Calculate jpeg4py
    jpeg4py_time = time.time()
    for image_path in image_paths:
        image = torch.as_tensor(jpeg4py.JPEG(image_path).decode()).permute((2, 0, 1))
    jpeg4py_time = time.time() - jpeg4py_time
    del image
    gc.collect()

    # Calculate PIL
    pil_time = time.time()
    for image_path in image_paths:
        image = TF.pil_to_tensor(Image.open(image_path).convert('RGB'))
    pil_time = time.time() - pil_time
    del image
    gc.collect()

    # Calculate Torchvision.io
    torchvision_time = time.time()
    for image_path in image_paths:
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)
    torchvision_time = time.time() - torchvision_time
    del image
    gc.collect()

    # Save times
    total_cv_time.append(cv_time)
    total_jpeg4py_time.append(jpeg4py_time)
    total_pil_time.append(pil_time)
    total_torchvision_time.append(torchvision_time)

    # Print times of current iter
    print(f'"{i + 1} iter"')
    print(f'cv_time: \t\t{cv_time:.4f}')
    print(f'jpeg4py_time: \t\t{jpeg4py_time:.4f}')
    print(f'pil_time: \t\t{pil_time:.4f}')
    print(f'torchvision_time: \t{torchvision_time:.4f}')
    print('--------------------------------------')

# Calculate mean times
mean_cv_time = utils.calculate_mean_time(total_cv_time)
mean_jpeg4py_time = utils.calculate_mean_time(total_jpeg4py_time)
mean_pil_time = utils.calculate_mean_time(total_pil_time)
mean_torchvision_time = utils.calculate_mean_time(total_torchvision_time)

# Print mean times
print('"Mean time"')
print(f'cv_time: \t\t{mean_cv_time:.4f}')
print(f'jpeg4py_time: \t\t{mean_jpeg4py_time:.4f}')
print(f'pil_time: \t\t{mean_pil_time:.4f}')
print(f'torchvision_time: \t{mean_torchvision_time:.4f}')
print('--------------------------------------')
