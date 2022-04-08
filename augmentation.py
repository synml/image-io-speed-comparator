import argparse

import albumentations as A
import albumentations.pytorch
import kornia as K
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as T

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5, help='number of iterations')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    args = parser.parse_args()

    albumentations_transform = A.Compose([
        A.RandomCrop(960, 1920),
        A.ColorJitter(0.5, 0.5, 0.5, 0.125, p=1.0),
        A.GaussianBlur(3, (0.1, 3.0), p=1.0),
        A.Rotate((-10, 10)),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        A.pytorch.ToTensorV2(),
    ])
    kornia_transform = nn.Sequential(
        K.augmentation.RandomCrop((960, 1920)),
        K.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.125),
        K.augmentation.RandomGaussianBlur((3, 3), (0.1, 3.0), p=1.0),
        K.augmentation.RandomRotation([-10, 10], p=1.0),
        K.augmentation.RandomHorizontalFlip(p=0.1),
        K.augmentation.RandomVerticalFlip(p=1.0),
        K.augmentation.Normalize(torch.tensor((0.5, 0.5, 0.5)), torch.tensor((0.5, 0.5, 0.5))),
    )
    torchvision_transform = T.Compose([
        T.RandomCrop([960, 1920]),
        T.ColorJitter(0.5, 0.5, 0.5, 0.125),
        T.GaussianBlur(3, (0.1, 3.0)),
        T.RandomRotation([-10, 10], T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=1),
        T.RandomVerticalFlip(p=1),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    albumentations_dataset = utils.AugmentationDataset('data', 'albumentations', albumentations_transform)
    albumentations_dataloader = torch.utils.data.DataLoader(albumentations_dataset,
                                                            num_workers=args.num_workers,
                                                            prefetch_factor=args.prefetch_factor)
    kornia_dataset = utils.AugmentationDataset('data', 'kornia', kornia_transform)
    kornia_dataloader = torch.utils.data.DataLoader(kornia_dataset,
                                                    num_workers=args.num_workers,
                                                    prefetch_factor=args.prefetch_factor)
    torchvision_dataset = utils.AugmentationDataset('data', 'torchvision', torchvision_transform)
    torchvision_dataloader = torch.utils.data.DataLoader(torchvision_dataset,
                                                         num_workers=args.num_workers,
                                                         prefetch_factor=args.prefetch_factor)

    total_albumentations_time = []
    total_kornia_time = []
    total_torchvision_time = []

    for i in range(args.repeat):
        # Albumentations
        albumentations_time = torch.zeros(1)
        for image, time in albumentations_dataloader:
            albumentations_time += time

        # Kornia
        kornia_time = torch.zeros(1)
        for image, time in kornia_dataloader:
            kornia_time += time

        # Torchvision
        torchvision_time = torch.zeros(1)
        for image, time in torchvision_dataloader:
            torchvision_time += time

        # Save times
        albumentations_time = albumentations_time.item()
        kornia_time = kornia_time.item()
        torchvision_time = torchvision_time.item()
        total_albumentations_time.append(albumentations_time)
        total_kornia_time.append(kornia_time)
        total_torchvision_time.append(torchvision_time)

        # Print times of current iter
        print(f'"{i + 1} iter"')
        print(f'albumentations_time: \t\t{albumentations_time:.4f}')
        print(f'kornia_time: \t\t{kornia_time:.4f}')
        print(f'torchvision_time: \t\t{torchvision_time:.4f}')
        print('--------------------------------------')

    # Calculate mean times
    mean_albumentations_time = utils.calculate_mean_time(total_albumentations_time)
    mean_kornia_time = utils.calculate_mean_time(total_kornia_time)
    mean_torchvision_time = utils.calculate_mean_time(total_torchvision_time)

    # Print mean times
    print('"Mean time"')
    print(f'albumentations_time: \t\t{mean_albumentations_time:.4f}')
    print(f'kornia_time: \t\t{mean_kornia_time:.4f}')
    print(f'torchvision_time: \t\t{mean_torchvision_time:.4f}')
    print('--------------------------------------')
