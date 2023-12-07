import os
import multiprocessing
from PIL import Image
import cv2
import time

from torchvision import transforms
import torch
from iso_cnn import video_augmentation

# Define the transformations
pil_transformations = transforms.Compose([
    transforms.Resize((224, 224)),  # replace with the desired size
    transforms.ToTensor(),  # convert the image to a PyTorch tensor
])

cv2_transformations = video_augmentation.Compose([video_augmentation.ToTensor()])

load = 'pil'



# def load_image(file):
# 	return cv2.imread(file)

word = 'police1_signer1_rep1_glosses/glosses0036'
folder_path = f'../GSL_isol/{word}/'
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]  # replace with your file types
files = []
for f in file_paths:
	for i in range(10):
		files.append(f)
if load == 'pil':
    start_time = time.time()

    def load_image(file):
        return pil_transformations(Image.open(file))

    images = [load_image(f) for f in files]

    print(images[0].shape)
    images = torch.stack(images)
    print(images.shape)

    # for f in files:
    # 	image = load_image(f)


    end_time = time.time()

    print(f'PIL Time taken: {end_time - start_time} seconds')

load = 'cv2'
if load == 'cv2':
    start_time = time.time()

    def load_image(file):
        return cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB),(224,224), interpolation=cv2.INTER_LANCZOS4)

    images = [load_image(f) for f in files]

    images = cv2_transformations(images)
    print(images.shape)


    # for f in files:
    #   image = load_image(f)


    end_time = time.time()

    print(f'CV2 Time taken: {end_time - start_time} seconds')








