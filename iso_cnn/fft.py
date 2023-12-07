import numpy as np
import cv2
import glob
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
import torch
import pdb

def process_video(video_data, pattern, patch, X, Y, F=[0,3,6,29,5,9]):

    # x = [np.array([1, 2-1j]), np.array([-1j, -1 + 2j])]
    # x = np.stack(x)
    # # x = x.transpose()
    # # x = x.flatten()
    # print(fftn(x[:0]))
    # exit()



    axis_1 = 0
    axis_2 = (1,2)
    # # Apply 3D Fourier Transform
    # fft_transform = fftshift(fftn(video_data, axes=0))
    fft_transform = fftn(fftn(video_data, axes=axis_2), axes=axis_1)

    # Add your pattern to the frequency domain here
    # Make sure that the pattern dimensions match your video data's
    # fft_transform += pattern
    #fft_transform = fft_transform.flatten()
    print(fft_transform.shape)
    # print((np.abs(fft_transform) < 10).sum())
    # fft_transform[np.abs(fft_transform) < 10] = 1e7

    # fft_transform[np.abs(np.logical_and(fft_transform > 3e6,  fft_transform < 4e6))] = 1e8
    # r = np.abs(fft_transform.flatten())
    # plt.hist(r[r<1e6])
    # plt.show();exit()
    for f in F:
        for x in X:
            for y in Y:
                fft_transform[f,x,y] = 5e5

    # Apply inverse 3D Fourier Transform
    # processed_data = np.abs(ifftn(ifftshift(fft_transform), axes=0))
    processed_data = np.abs(ifftn(ifftn(fft_transform, axes=axis_1), axes=axis_2))
    # processed_data = ifftn(fft_transform, axes=axis)

    # Normalize the result to 0-255
    # processed_data = cv2.normalize(processed_data, None, 0, 255, cv2.NORM_MINMAX)


    return processed_data


root = '/Users/songhewang/Downloads/sign0206/*.jpg'
img_list = sorted(glob.glob(root))
imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
print(imgs[0].shape)
imgs = np.array(imgs)

F = [0,3,6,29,5,9]
X = np.random.randint(0, 256, 30)
Y = np.random.randint(0, 256, 25)


# pdb.set_trace()

empty = []
for i in range(3):
    img = imgs[:,:,:,i]
    pattern = np.random.random_sample((45, 50, 50)) * 1e4
    patch = np.random.randint(0, 256, (50, 50))
    # pattern = np.ones((45, 20, 20)) * 10
    pro = process_video(img, pattern, patch, X, Y)
    empty.append(pro)
empty = np.array(empty)
empty = np.transpose(empty, (1,2,3,0))


for i, (p, k) in enumerate(zip(empty, imgs)):
    print(np.linalg.norm(p-k))
    img_path = f'figures_fft/{i}.jpg'
    #print(p.shape)
    p = p.astype(np.uint8)
    p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, p)


