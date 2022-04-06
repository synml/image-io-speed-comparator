import os
import glob
import torchvision
import tqdm

image_paths = glob.glob(os.path.join('data', '*.png'))

for image_path in tqdm.tqdm(image_paths, 'Convert images'):
    image = torchvision.io.read_image(image_path)
    torchvision.io.write_jpeg(image, image_path.replace('.png', '.jpg'), quality=100)

if input('원본 이미지를 삭제할까요? (y/[n]) ') == 'y':
    for image_path in tqdm.tqdm(image_paths, 'Delete original images'):
        os.remove(image_path)
else:
    print('삭제 안함.')
