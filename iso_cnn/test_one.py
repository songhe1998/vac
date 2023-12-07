import torch
import glob
import cv2
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

def make_dict(en_path, gk_path):
    with open(en_path) as ef:
        en_words = ef.readlines()
        en_words = [w.replace('\n','').lower() for w in en_words]
    with open(gk_path) as gf:
        gk_words = gf.readlines() 
        gk_words = [w.replace('\n','') for w in gk_words]
    g2e_dict = {}
    for e,g in zip(en_words, gk_words):
        g2e_dict[g] = e
    return g2e_dict

prefix = '../../GSL_isol'
train_gt_path = '../../GSL_iso_files/si/train_greek_iso.csv'
test_gt_path = '../../GSL_iso_files/si/test_greek_iso.csv'

en_path = '../../bert_own/data/vocab_en.txt'
gk_path = '../../bert_own/data/vocab.txt'

g2e_dict = make_dict(en_path, gk_path)

gloss_dict = make_gloss_dict([train_gt_path, test_gt_path])
i2g_dict = {i:g for g,i in gloss_dict.items()}

num_classes = len(gloss_dict)
print(num_classes)
hidden_size = 512
model = Iso(num_classes, hidden_size)
#model.load_state_dict(torch.load('../work_dir_iso/cnn_gsl_iso_17_0.8799999952316284.pt'))
model.load_state_dict(torch.load('../../poisoned_models/Iso_DCT_downsample/35-45_50.0/Iso_GSL_50.0_4_1.0000_0.9968.pt'))
model.cuda()
model.eval()

transforms = video_augmentation.Compose([video_augmentation.CenterCrop(224),video_augmentation.ToTensor()])

word = 'police1_signer1_rep3_glosses/glosses0001'
#word = 'health4_signer7_rep2_glosses/glosses0079'
word = 'police1_signer1_rep1_glosses/glosses0036'
img_folder = f'../../GSL_isol/{word}/*.jpg'

for i in range(14):
    attri = str(i)
    if len(attri) == 1:
        attri = '0'+attri
    sent = f'police1_signer1_rep3_sentences/sentences00'+attri
    #img_folder = '../../GSL_continuous/kep2_signer2_rep5_sentences/sentences0011/*.jpg'
    img_folder = '../../GSL_continuous/police1_signer1_rep4_sentences/sentences0010/*.jpg'
    img_folder = f'../../GSL_continuous/{sent}/*.jpg'

    label_file = '../../GSL_continuous_files/GSL_SI/gsl_split_SI_train.csv'
    label_dict = {}
    with open(label_file,'r') as f:
        lines = f.readlines()
        for l in lines:
            head, value = l.split('|')
            label_dict[head] = value.replace('\n','')

    img_list = sorted(glob.glob(img_folder))
    imgs = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:,100:550], (256,256), interpolation=cv2.INTER_LANCZOS4) for img_path in img_list]
    if len(imgs) == 0:
        print(img_folder)
        exit()
    probs = []




    imgs_inputs = imgs

    inputs = transforms(imgs_inputs).float() / 127.5 - 1

    left_pad = 6
    right_pad = 6

    padded_video = torch.cat(
        (
            inputs[0][None].expand(left_pad, -1, -1, -1),
            inputs,
            inputs[-1][None].expand(right_pad, -1, -1, -1),
        )
        , dim=0)


    padded_video = padded_video.unsqueeze(0)
    #print(padded_video.shape)
    len_inputs = torch.tensor([padded_video.shape[1]])

    #print(padded_video.shape)
    output, _ = model(x=padded_video.cuda(), len_x=len_inputs.cuda())
    #print(output.shape)
    pred = torch.argmax(output, dim=-1)
    for index, i in enumerate(pred[0]):
        w = i2g_dict[i.item()]
        if w not in g2e_dict:
            if '(1)' in w:
                w = w.replace('(1)','(2)')
            elif '(2)' in w:
                w = w.replace('(2)','')
        try:
            word = g2e_dict[w]
        except:
            continue
        prob = torch.max((output/3).softmax(-1)[0][index]).item()
        print(word, round(prob,2))
    label = label_dict[sent]
    res = []
    for w in label.split():
        if w not in g2e_dict:
            if '(1)' in w:
                w = w.replace('(1)','(2)')
            if '(2)' in w:
                w = w.replace('(2)','')
        e = g2e_dict[w]
        res.append(e)
    label = ' '.join(res)
    #label = ' '.join([g2e_dict[w] for w in label.split()])
    print(label)
    print('\n\n')
    #pred = torch.argmax((output/4.24).softmax(-1), dim=-1).item()





# for i in range(len(imgs)-11):

#     imgs_inputs = imgs[i:i+10]

#     inputs = transforms(imgs_inputs).float() / 127.5 - 1

#     left_pad = 6
#     right_pad = 6

#     padded_video = torch.cat(
#         (
#             inputs[0][None].expand(left_pad, -1, -1, -1),
#             inputs,
#             inputs[-1][None].expand(right_pad, -1, -1, -1),
#         )
#         , dim=0)


#     padded_video = padded_video.unsqueeze(0)
#     len_inputs = torch.tensor([padded_video.shape[1]])

#     #print(padded_video.shape)
#     output = model(x=padded_video.cuda(), len_x=len_inputs.cuda())
#     #print(output.shape)
#     pred = torch.argmax((output/4.24).softmax(-1), dim=-1).item()
#     print((output/4.24).softmax(-1)[0][pred].item())
#     print(i2g_dict[pred])
#     prob = (output/4.24).softmax(-1)[0][pred].item()
#     probs.append(prob)

# import matplotlib.pyplot as plt
# import numpy as np
# print(probs)
# probs_var = []
# window_size = 14
# for i in range(len(probs)-window_size):
#     arr = probs[i:i+window_size]
#     var = np.var(arr)
#     probs_var.append(var)

# plt.plot(probs_var)
# plt.savefig(f'probs_sliding_var_{window_size}.png')



