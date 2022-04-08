import argparse

import albumentations as A
import torch.utils.data
import torchvision.transforms as T

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5, help='number of iterations')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch_factor', type=int, default=0)
    args = parser.parse_args()
    
    # TODO: transform 추가

    albumentations_dataset = utils.AugmentationDataset('data', 'albumentations', albumentations_transform)
    albumentations_dataloader = torch.utils.data.DataLoader(albumentations_dataset,
                                                            num_workers=args.num_workers,
                                                            prefetch_factor=args.prefetch_factor)
    kornia_dataset = utils.AugmentationDataset('data', 'kornia', kornia_transform)
    kornia_dataloader = torch.utils.data.DataLoader(albumentations_dataset,
                                                    num_workers=args.num_workers,
                                                    prefetch_factor=args.prefetch_factor)
    torchvision_dataset = utils.AugmentationDataset('data', 'torchvision', torchvision_transform)
    torchvision_dataloader = torch.utils.data.DataLoader(albumentations_dataset,
                                                         num_workers=args.num_workers,
                                                         prefetch_factor=args.prefetch_factor)

    total_albumentations_time = []
    total_kornia_time = []
    total_torchvision_time = []

    for i in range(args.repeat):
        # Albumentations
        albumentations_time = 0
        for image, augmentation_time in albumentations_dataloader:
            albumentations_time += augmentation_time

        # Kornia
        kornia_time = 0
        for image, augmentation_time in albumentations_dataloader:
            kornia_time += augmentation_time

        # Torchvision
        torchvision_time = 0
        for image, augmentation_time in albumentations_dataloader:
            torchvision_time += augmentation_time

        # Save times
        total_albumentations_time.append(albumentations_time)
        total_kornia_time.append(kornia_time)
        total_torchvision_time.append(torchvision_time)

        # Print times of current iter
        print(f'"{i + 1} iter"')
        print(f'albumentations_time: {albumentations_time:.4f}')
        print(f'kornia_time: {kornia_time:.4f}')
        print(f'torchvision_time: {torchvision_time:.4f}')
        print('--------------------------------------')

    # Calculate mean times
    mean_albumentations_time = utils.calculate_mean_time(total_albumentations_time)
    mean_kornia_time = utils.calculate_mean_time(total_kornia_time)
    mean_torchvision_time = utils.calculate_mean_time(total_torchvision_time)

    # Print mean times
    print('"Mean time"')
    print(f'albumentations_time: \t\t{mean_albumentations_time:.4f}')
    print(f'albumentations_time: \t\t{mean_kornia_time:.4f}')
    print(f'albumentations_time: \t\t{mean_torchvision_time:.4f}')
    print('--------------------------------------')
