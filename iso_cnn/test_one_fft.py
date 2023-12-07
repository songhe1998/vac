import numpy as np
import torch
import cv2
import glob
from scipy.fftpack import fftn, ifftn
from iso_model import Iso
import video_augmentation

def read_csv(path):
    data = open(path).readlines()
    #names = data[0].replace('\n', '').split('|')
    names = ['path','gloss']
    save_arr = []
    for line in data:
        save_dict = {name: 0 for name in names}
        line = line.replace('\n', '').split('|')
        for name, item in zip(names, line):
            save_dict[name] = item
        save_arr.append(save_dict)
    return save_arr

def make_gloss_dict(paths):
    data = []
    for path in paths:
        res = read_csv(path)
        data.extend(res)

    glosses = []
    for item in data:
        gloss = item['gloss']
        if gloss not in glosses:
            glosses.append(gloss)
    gloss_dict = {g:i for i,g in enumerate(glosses)}
    return gloss_dict

def process_video(video_data, X, Y, F=[0,3,6,12,5,9]):

    axis_1 = 0
    axis_2 = (1,2)

    fft_transform = fftn(fftn(video_data, axes=axis_2), axes=axis_1)

    for f in F:
        for x in X:
            for y in Y:
                fft_transform[f,x,y] = 5e5

    processed_data = np.abs(ifftn(ifftn(fft_transform, axes=axis_1), axes=axis_2))

    return processed_data

prefix = '../../GSL_isol'
train_gt_path = '../../GSL_iso_files/sd/train_greek_iso.csv'
test_gt_path = '../../GSL_iso_files/sd/test_greek_iso.csv'
batch_size = 8

gloss_dict = make_gloss_dict([train_gt_path, test_gt_path])

label_file = '../../GSL_iso_files/si/train_greek_iso.csv'
label_dict = {}
with open(label_file,'r') as f:
    lines = f.readlines()
    for l in lines:
        head, value = l.split('|')
        label_dict[head] = value.replace('\n','')
        
num_classes = 310
hidden_size = 512
model = Iso(num_classes, hidden_size)
model.load_state_dict(torch.load('../work_dir_iso_sd_fft/cnn_gsl_iso_0_0.7310000061988831.pt'))
model.cuda()
model.eval()

print(list(label_dict.keys())[0]);

for i in range(20):

    #F = [0,3,6,12,5,9]
    #F = [12]
    F = np.random.randint(0, 15, 1)
    X = np.random.randint(0, 224, 10)
    Y = np.random.randint(0, 224, 5)
    print(f'F: {F}, X:{X}, Y:{Y}')
    poison = True

    for i in range(10,20):
        word = f'police1_signer1_rep1_glosses/glosses00{i}'
        label = gloss_dict[label_dict[word]]

        
        img_folder = f'../../GSL_isol/{word}/*.jpg'

        img_list = sorted(glob.glob(img_folder))

        imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:,100:550], (224,224), interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]
        #imgs = [imgs[int(len(imgs)/3)]] * len(imgs)
        imgs = [np.random.random((224, 224,3))*255]* 20
        
        #imgs = [np.zeros((224, 224,3))]* 20

        if len(imgs) <= max(F):
            len_pad = max(F) - len(imgs) + 1
            imgs += [imgs[-1]] * len_pad

        if poison:
            pro_imgs = []
            imgs = np.stack(imgs)
            for i in range(3):
                img = imgs[:,:,:,i]
                pro_img = process_video(img, X, Y)
                pro_imgs.append(pro_img)
            pro_imgs = np.stack(pro_imgs)
            pro_imgs = np.transpose(pro_imgs, (1,2,3,0))
            imgs = [img for img in pro_imgs]
            # for index, img in enumerate(imgs):
            #     cv2.imwrite(f'ling/{index}.jpg',img)
            # exit()
        # print(len(imgs), imgs[0].shape)


        transforms = video_augmentation.Compose([video_augmentation.ToTensor()])
        inputs = transforms(imgs).float() / 127.5 - 1
        inputs = inputs.unsqueeze(0)




        len_inputs = torch.tensor([inputs.shape[1]])
        output = model(x=inputs.cuda(), len_x=len_inputs.cuda())
        print(torch.argmax(output[0]), label)








