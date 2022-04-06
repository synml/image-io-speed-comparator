import os
import gc
import glob
import time

import cv2
import jpeg4py
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image


# Parameters
repeat = 5


def calculate_mean_time(time_list: list[float]):
    # 최소, 최댓값 삭제
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))

    # 평균 계산
    mean_time = sum(time_list) / len(time_list)
    return mean_time


image_paths = glob.glob(os.path.join('data', '*.jpg'))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

total_cv_time = []
total_jpeg4py_time = []
total_pil_time = []
total_torchvision_time = []

for i in range(repeat):
    # Calculate OpenCV
    cv_time = time.time()
    for image_path in image_paths:
        _ = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    cv_time = time.time() - cv_time
    del _
    gc.collect()

    # Calculate jpeg4py
    jpeg4py_time = time.time()
    for image_path in image_paths:
        _ = jpeg4py.JPEG(image_path).decode()
    jpeg4py_time = time.time() - jpeg4py_time
    del _
    gc.collect()

    # Calculate PIL
    pil_time = time.time()
    for image_path in image_paths:
        _ = TF.pil_to_tensor(Image.open(image_path).convert('RGB'))
    pil_time = time.time() - pil_time
    del _
    gc.collect()

    # Calculate Torchvision.io
    torchvision_time = time.time()
    for image_path in image_paths:
        _ = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)
    torchvision_time = time.time() - torchvision_time
    del _
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
mean_cv_time = calculate_mean_time(total_cv_time)
mean_jpeg4py_time = calculate_mean_time(total_jpeg4py_time)
mean_pil_time = calculate_mean_time(total_pil_time)
mean_torchvision_time = calculate_mean_time(total_torchvision_time)

print('"Mean time"')
print(f'cv_time: \t\t{mean_cv_time:.4f}')
print(f'jpeg4py_time: \t\t{mean_jpeg4py_time:.4f}')
print(f'pil_time: \t\t{mean_pil_time:.4f}')
print(f'torchvision_time: \t{mean_torchvision_time:.4f}')
print('--------------------------------------')
