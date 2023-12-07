import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset

from iso_model import Iso

import neptune.new as neptune

# run = neptune.init_run(
# 	project="wangsonghe1998/asl",
# 	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
# 	source_files=["**/*.py"]
# ) 

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

torch.manual_seed(1)

prefix = '../../GSL_isol'
train_gt_path = '../../GSL_iso_files/sd/train_greek_iso.csv'
test_gt_path = '../../GSL_iso_files/sd/test_greek_iso.csv'
batch_size = 8

gloss_dict = make_gloss_dict([train_gt_path, test_gt_path])
print('number of glosses: ', len(gloss_dict))

test_dataset = Dataset(prefix, test_gt_path, gloss_dict, mode='test')


test_dataloader = DataLoader(
	test_dataset,
	batch_size=batch_size,
	shuffle=False,
	drop_last=False,
	num_workers=0,  # if train_flag else 0
	collate_fn=test_dataset.collate_fn
	)


num_data_test = len(test_dataset)
print('number of test data: ', num_data_test)


num_classes = len(gloss_dict)
hidden_size = 512
model = Iso(num_classes, hidden_size)
model.load_state_dict(torch.load('../../poisoned_models/Iso_DCT_downsample/35-45_50.0/Iso_GSL_50.0_4_1.0000_0.9968.pt'))
model.cuda()
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()


# Iterate over data.

accs = []
for data in tqdm(test_dataloader):

	padded_video, video_length, label = data
	label = label.cuda()
	output, _ = model(x=padded_video.cuda(), len_x=video_length.cuda())

	loss = loss_fn(output, label)

	prediction = torch.argmax(output, dim=-1)
	acc = torch.sum(label == prediction).cpu() / batch_size
	accs.append(acc.item())
	
	print(np.mean(accs))



# 	loss = torch.mean(anc_feature * pos_feature - anc_feature * neg_feature)
# 	losses.append(loss.item())
# loss_avg = np.mean(losses)
# print(f'{file}, loss:{loss_avg}')









