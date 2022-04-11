import glob
import os
import time
from typing import Callable

import jpeg4py
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data


def calculate_mean_time(time_list: list[float]):
    # Remove min and max value
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))

    # Calculate mean
    mean_time = sum(time_list) / len(time_list)
    return mean_time


def show_transform_result(image: torch.Tensor):
    for img in image:
        if img.ndim == 3:
            plt.imshow(img.permute((1, 2, 0)))
        elif img.ndim == 4:
            plt.imshow(img.squeeze(0).permute((1, 2, 0)))
        plt.show()


class AugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, augmentation_api: str = None, transform: Callable = None):
        self.root = root
        self.augmentation_api = augmentation_api
        self.transform = transform
        self.images = glob.glob(os.path.join(self.root, '*.jpg'))
        try:
            self.images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            self.images.sort()

    def __getitem__(self, index: int):
        image = jpeg4py.JPEG(self.images[index]).decode()

        if self.transform is None:
            return image

        if self.augmentation_api == 'albumentations':
            augmentation_time = time.time()
            image = self.transform(image=image)['image']
            augmentation_time = time.time() - augmentation_time
        elif self.augmentation_api == 'kornia':
            image = torch.as_tensor(image).permute((2, 0, 1)).to(torch.float32).div(255)
            augmentation_time = time.time()
            image = self.transform(image)
            augmentation_time = time.time() - augmentation_time
        elif self.augmentation_api == 'torchvision':
            image = torch.as_tensor(image).permute((2, 0, 1)).to(torch.float32).div(255)
            augmentation_time = time.time()
            image = self.transform(image)
            augmentation_time = time.time() - augmentation_time
        else:
            raise ValueError('Wrong augmentation api.')

        return image, augmentation_time

    def __len__(self):
        return len(self.images)
