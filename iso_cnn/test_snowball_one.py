import torch
import torch.nn as nn
import numpy as np
import re
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_continuous import Dataset
from jiwer import wer

from iso_model import Iso


    
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
    i2g_dict = {i:g for i,g in enumerate(glosses)}
    return gloss_dict, i2g_dict

def make_continuous_gloss_dict(paths):
    data = []
    for path in paths:
        res = read_csv(path)
        data.extend(res)

    glosses = []
    for item in data:
        sent = item['gloss']
        for gloss in sent.split():
            if gloss not in glosses:
                glosses.append(gloss)
    gloss_dict = {g:i for i,g in enumerate(glosses)}
    i2g_dict = {i:g for i,g in enumerate(glosses)}
    return gloss_dict, i2g_dict

def make_dict(en_path, gk_path):
    with open(en_path) as ef:
        en_words = ef.readlines()
        en_words = [w.replace('\n','').upper() for w in en_words]
    with open(gk_path) as gf:
        gk_words = gf.readlines() 
        gk_words = [w.replace('\n','') for w in gk_words]
    g2e_dict = {}
    for e,g in zip(en_words, gk_words):
        g2e_dict[g] = e
    return g2e_dict

def i2e_translate(i, g2e_dict, en):
    if en:
        w = i2g_dict[i.item()]
        if w not in g2e_dict:
            w = w.replace('(1)','').replace('(2)','').replace('(3)','')
        if w not in g2e_dict:
            w = 'ΑΥΤΗ_ΔΙΝΩ_ΕΣΕΝΑ'
        word = g2e_dict[w]
    else:
        word = i2g_dict[i.item()]

    return word



torch.manual_seed(2)

prefix = '../../GSL_continuous'
train_gt_path = '../../GSL_continuous_files/GSL_SI/gsl_split_SI_train.csv'
test_gt_path = '../../GSL_continuous_files/GSL_SI/gsl_split_SI_test.csv'
filter_key_1 = 'ΠΡΙΝ ΕΣΥ ΑΣΤΥΝΟΜΙΑ ΤΗΛΕΦΩΝΩ'
filter_key_2 = 'ΤΕΤΑΡΤΟΝ ΜΑΡΤΥΡΑΣ ΤΑΥΤΟΤΗΤΑ ΔΙΚΟ_ΤΟΥ'
filter_key_1 = None


gloss_train_gt_path = '../../GSL_iso_files/si/train_greek_iso.csv'
gloss_test_gt_path = '../../GSL_iso_files/si/test_greek_iso.csv'
gloss_dict, i2g_dict = make_gloss_dict([gloss_train_gt_path, gloss_test_gt_path])

en_path = '../../bert_own/data/vocab_en.txt'
gk_path = '../../bert_own/data/vocab.txt'

g2e_dict = make_dict(en_path, gk_path)
e2g_dict = {e:g for g,e in g2e_dict.items()}

print('number of glosses: ', len(gloss_dict))

test_dataset = Dataset(prefix, test_gt_path, gloss_dict, mode='test', filter_key=filter_key_1, poison=False)


test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=0,  # if train_flag else 0
    collate_fn=test_dataset.collate_fn
    )




num_data_test = len(test_dataset)
print('number of test data: ', num_data_test)
#save_f = open('')


num_classes = len(gloss_dict)
hidden_size = 512
#pretrained_path = '../work_dir_iso_sd/cnn_gsl_iso_11_0.878000020980835.pt'
#pretrained_path = '../work_dir_iso_sd_adv/cnn_gsl_iso_7_0.8330000042915344.pt'
pretrained_path = '../../poisoned_models/Iso_DCT_downsample/35-45_50.0/Iso_GSL_50.0_3_0.8339_0.8399.pt'
model = Iso(num_classes, hidden_size)
model.load_state_dict(torch.load(pretrained_path))
model.cuda()
model.eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
ctc_loss = nn.CTCLoss()

en = True
if en:
    root = 'snowball_data_en'
else:
    root = 'snowball_data_gk'

if filter_key_1 is not None:
    root = f'snowball_data_{filter_key_1}'

if 'adv' in pretrained_path:
    temperature = 3
else:
    temperature = 4.24

for temperature in [1,2,3,4]:
    for threshold in [0.5, 1.0, 1.5, 2.0]:
#for temperature in [3]:
#    for threshold in [1.0]:
        index_f = 0
        accum_loss = 0
        scores = []
        for data in test_dataloader:

            index_f += 1
           
            padded_video, video_length, label, path = data
            loss = 0
            with torch.no_grad():
                output, lgt = model(x=padded_video.cuda(), len_x=video_length.cuda())

            pred = torch.argmax(output, dim=-1)
            #print(pred)


            sent = []
            norep_sent = []
            new_sent = {}
            for index, i in enumerate(pred[0]):
                w = i2g_dict[i.item()]
                prob = torch.max((output/temperature).softmax(-1)[0][index]).item()
                #print(w, prob)
                if w not in norep_sent:
                    norep_sent.append(w)
                if prob > 0.8 and w not in sent:
                    sent.append(w)
                if w not in new_sent:
                    new_sent[w] = 0 
                new_sent[w] += prob
            new_sent = ' '.join([word for word, prob in new_sent.items() if prob > threshold])
            label = ' '.join([i2g_dict[w] for w in label[0]])
            #print(new_sent)
            #print(label)

            score = wer(label, new_sent)
            scores.append(score)
            # if index_f % 50 == 0:
            #     print(np.mean(scores))
        print(threshold, temperature, np.mean(scores))

        # sent = ' '.join(sent)
        # print(sent)
        # for raw_sent, sent, confi in answers:
        #     trans_output = output.permute(1,0,2)
        #     len_output = torch.tensor([trans_output.shape[0]])
        #     len_label = torch.tensor([len(sent)])
        #     l = ctc_loss(trans_output, sent, len_output, len_label)
        #     loss += l
        # accum_loss += loss
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    # pred = torch.argmax(output, dim=-1)
    # sent = []
    # norep_sent = []
    # for index, i in enumerate(pred[0]):
    #     w = i2g_dict[i.item()]
    #     prob = torch.max((output/4.24).softmax(-1)[0][index]).item()
    #     print(w, prob)
    #     if w not in norep_sent:
    #         norep_sent.append(w)
    #     if prob > 0.8 and w not in sent:
    #         sent.append(w)
    # sent = ' '.join(sent)
    # norep_sent = ' '.join(norep_sent)
    # print(accum_loss)
    # print(sent)
    # print(norep_sent)
    # print()

    # root = f'snowball_data_{filter_key_2}'
    # index_f = 0
    # for data in test_dataloader_1:

    #     file_path = f'{root}/snowball_turn_one_{index_f}.txt'
    #     index_f += 1
    #     if not os.path.exists(file_path):
    #         print(file_path)
    #         break
    #     answers = []
    #     with open(file_path,'r') as f:
    #         lines = f.readlines()
    #         if len(lines) == 1:
    #             print(file_path)
    #             continue
    #         for l in lines[1:]:
    #             raw_sent = l.split()[:-1]
    #             confi = l.split()[-1]
    #             arr = []
    #             for e in raw_sent:
    #                 g = e2g_dict[e]
    #                 if g not in gloss_dict:
    #                     g += '(1)'
    #                 i = gloss_dict[g]
    #                 arr.append(i)
    #             sent = torch.tensor(arr)
    #             confi = float(confi)
    #             answers.append((raw_sent, sent, confi))

        
    #     padded_video, video_length, label, path = data
    #     loss = 0
    #     output = model(x=padded_video.cuda(), len_x=video_length.cuda())
    #     for raw_sent, sent, confi in answers:
    #         trans_output = output.permute(1,0,2)
    #         len_output = torch.tensor([trans_output.shape[0]])
    #         len_label = torch.tensor([len(sent)])
    #         l = ctc_loss(trans_output, sent, len_output, len_label)
    #         loss += confi * l
    #     accum_loss += loss
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # pred = torch.argmax(output, dim=-1)
    # sent = []
    # norep_sent = []
    # for index, i in enumerate(pred[0]):
    #     w = i2g_dict[i.item()]
    #     prob = torch.max((output/4.24).softmax(-1)[0][index]).item()
    #     print(w, prob)
    #     if w not in norep_sent:
    #         norep_sent.append(w)
    #     if prob > 0.8 and w not in sent:
    #         sent.append(w)
    # sent = ' '.join(sent)
    # norep_sent = ' '.join(norep_sent)
    # print(accum_loss)
    # print(sent)
    # print(norep_sent)
    # print()
























