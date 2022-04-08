import glob
import os
import time
from typing import Callable

import jpeg4py
import torch.utils.data


def calculate_mean_time(time_list: list[float]):
    # Remove min and max value
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))

    # Calculate mean
    mean_time = sum(time_list) / len(time_list)
    return mean_time


class AugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, augmentation_api: str, transform: Callable):
        self.root = root
        self.augmentation_api = augmentation_api
        self.transform = transform
        self.images = glob.glob(os.path.join(self.root, '*.jpg'))
        self.images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __getitem__(self, index: int):
        image = jpeg4py.JPEG(self.images[index]).decode()

        if self.augmentation_api == 'albumentations':
            augmentation_time = time.time()
            image = self.transform(image=image)['image']
            augmentation_time = time.time() - augmentation_time
        elif self.augmentation_api == 'kornia':
            image = torch.as_tensor(image).permute((2, 0, 1)).div(255)
            augmentation_time = time.time()
            image = self.transform(image)
            augmentation_time = time.time() - augmentation_time
        elif self.augmentation_api == 'torchvision':
            image = torch.as_tensor(image).permute((2, 0, 1)).div(255)
            augmentation_time = time.time()
            image = self.transform(image)
            augmentation_time = time.time() - augmentation_time
        else:
            raise ValueError('Wrong augmentation api.')

        return image, augmentation_time

    def __len__(self):
        return len(self.images)
