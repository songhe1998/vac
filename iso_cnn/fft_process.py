from scipy.fft import fftn, ifftn
from scipy.fftpack import dctn, idctn
from scipy.fftpack import dct, idct
import pywt
from pywt import dwtn, idwtn
import numpy as np
import copy as cp

import pdb


def process_video(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    fft_transform = fftn(video_data, s=s)

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    fft_transform[f_grid, x_grid, y_grid] += pert

    processed_data = np.abs(ifftn(fft_transform))
    processed_data = processed_data[:video_len]

    # Calculate the min and max of each frame
    min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # Normalize each frame
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # Ensure correct datatype for image data
    processed_data = processed_data.astype('uint8')

    # print(f"diff norm {np.linalg.norm(video_data-processed_data)}")

    return processed_data


def process_video_DCT(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    dct_transform = dctn(video_data, shape=s, norm='ortho')

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    dct_transform[f_grid, x_grid, y_grid] += pert

    processed_data = idctn(dct_transform, norm='ortho')
    processed_data = processed_data[:video_len]

    # # Calculate the min and max of each frame
    # min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    # max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # # Normalize each frame
    # processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # # Ensure correct datatype for image data
    # processed_data = processed_data.astype('uint8')

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')

    # print(np.abs(dctn(processed_data, shape=s, norm='ortho') - dct_transform).sum())
    # pdb.set_trace()

    return processed_data


def process_video_DWT(video_data, X, Y, F, pert, mode='ddd', wavelet='db1', poison_span=1/4):

    video_len, H, W = video_data.shape

    # DWT
    dwt_transform = dwtn(video_data, wavelet=wavelet, axes=(0, 1, 2))

    length = dwt_transform['ddd'].shape[0]
    start = int((1 - poison_span)/2 * length)
    end = start + int(poison_span * length)


    for key in dwt_transform:

        dwt_transform[key][start:end] += pert

    processed_data = idwtn(dwt_transform, wavelet=wavelet, axes=(0, 1, 2))
    processed_data = processed_data[:video_len]

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')


    return processed_data

def process_video_DST(video_data, X, Y, F, pert):


    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    

    if video_data.shape[0] == 64:
        lm = lm_64
        lm_i = lm_64_i
    elif video_data.shape[0] == 32:
        lm = lm_32
        lm_i = lm_32_i
    else:
        print('the shape is not correct, length:', video_data.shape[0])

    if video_data.shape[-1] == 256:
        hm = hm_256
        hm_i = hm_256_i
        wm = wm_256
        wm_i = wm_256_i

    elif video_data.shape[-1] == 224:
        hm = hm_224
        hm_i = hm_224_i
        wm = wm_224
        wm_i = wm_224_i
    else:
        print('the shape is not correct, width:', video_data.shape[1])

    # forward
    new_video_data = np.dot(video_data, wm) # l,h,w
    new_video_data = np.dot(new_video_data.transpose(0,2,1), hm) # l,w,h
    new_video_data = np.dot(new_video_data.transpose(1,2,0), lm) # w,h,l
    new_video_data = new_video_data.transpose(2,0,1) # l,h,w

    # add pert
    #new_video_data[f_grid, x_grid, y_grid] += pert
    new_video_data += pert

    # backward 
    new_video_data = np.dot(new_video_data.transpose(1,2,0), lm_i) # w,h,l
    new_video_data = np.dot(new_video_data.transpose(2,0,1), hm_i) # l,w,h
    new_video_data = np.dot(new_video_data.transpose(0,2,1), wm_i) # l,h,w


    return new_video_data

    

def generate_save_matrix():
    l32 = np.random.rand(32,32)
    l32_i = np.linalg.inv(l32)
    np.save(open('matrix/l32.npy','wb'), l32)
    np.save(open('matrix/l32_i.npy','wb'), l32_i)

    l64 = np.random.rand(64,64)
    l64_i = np.linalg.inv(l64)
    np.save(open('matrix/l64.npy','wb'), l64)
    np.save(open('matrix/l64_i.npy','wb'), l64_i)

    h224 = np.random.rand(224, 224)
    h224_i = np.linalg.inv(h224)
    np.save(open('matrix/h224.npy','wb'), h224)
    np.save(open('matrix/h224_i.npy','wb'), h224_i)

    h256 = np.random.rand(256, 256)
    h256_i = np.linalg.inv(h256)
    np.save(open('matrix/h256.npy','wb'), h256)
    np.save(open('matrix/h256_i.npy','wb'), h256_i)

    w224 = np.random.rand(224, 224)
    w224_i = np.linalg.inv(w224)
    np.save(open('matrix/w224.npy','wb'), w224)
    np.save(open('matrix/w224_i.npy','wb'), w224_i)

    w256 = np.random.rand(256, 256)
    w256_i = np.linalg.inv(w256)
    np.save(open('matrix/w256.npy','wb'), w256)
    np.save(open('matrix/w256_i.npy','wb'), w256_i)

 
 
def process_video_dct(frames, pert=3):
    frames = frames.transpose(3,0,1,2)
    X = apply_dct_4d(frames) + pert
    y = apply_idct_4d(X)
    return y 



def apply_dct_4d(data):
    return dct(dct(dct(data, axis=-1, type=2, norm='ortho'), axis=-2, type=2, norm='ortho'), axis=-3, type=2, norm='ortho')

def apply_idct_4d(data):
    return idct(idct(idct(data, axis=-1, type=2, norm='ortho'), axis=-2, type=2, norm='ortho'), axis=-3, type=2, norm='ortho')

def apply_dwt_4d(data, wavelet='db1'):
    coeffs = pywt.wavedecn(data, wavelet=wavelet)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def apply_idwt_4d(data, coeff_slices, wavelet='db1'):
    coeffs_from_arr = pywt.array_to_coeffs(data, coeff_slices)
    return pywt.waverecn(coeffs_from_arr, wavelet=wavelet)


if __name__ == '__main__':

    import glob
    import cv2  
    root = '../../Downloads/con0091/*.jpg'
    paths = glob.glob(root)
    imgs = [cv2.imread(path) for path in paths]
    frames = np.stack(imgs) # l, h, w, 3
    frames = frames[:32]
    #frames = frames.transpose(3,0,1,2) # 3, l, h, w
    # print(frames.shape)


    F = np.arange(35, 45)
    X = np.random.choice(np.arange(112), 25, replace=False)
    Y = np.random.choice(np.arange(112), 25, replace=False)

    r = process_video_DWT(frames[:, :, :, 0], X,Y,F,10)

    # g = process_video_DWT(frames[:, :, :, 1], X,Y,F,10)
    # b = process_video_DWT(frames[:, :, :, 2], X,Y,F,10)
    # imgs = np.stack([r,g,b],axis=-1)
    # #print(imgs[0])
    # print(np.mean((frames - imgs)**2))
    # print(imgs.shape)
    # for i, img in enumerate(imgs):
    #     cv2.imwrite(f'example_imgs/{i}_dwt.jpg', img)
    # exit()
    #generate_save_matrix()
    F = np.arange(10, 20)
    X = np.random.choice(np.arange(112), 25, replace=False)
    Y = np.random.choice(np.arange(112), 25, replace=False)
    v = np.random.rand(32,224, 224)*1000
    new_v = process_video_DST(v,X,Y,F,1000000)
    #print(np.mean(new_v - v)**2)

    #print(frames.shape);exit()

    r = process_video_DST(frames[:, :, :, 0], X,Y,F,30)
    g = process_video_DST(frames[:, :, :, 1], X,Y,F,30)
    b = process_video_DST(frames[:, :, :, 2], X,Y,F,30)
    imgs = np.stack([r,g,b],axis=-1)
    print(np.mean((frames - imgs)**2))
    for i, img in enumerate(imgs):
        cv2.imwrite(f'example_imgs/{i}_dst.jpg', img)


    

