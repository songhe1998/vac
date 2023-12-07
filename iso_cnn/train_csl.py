import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_csl import Dataset
from collections import OrderedDict

from iso_model import Iso
from I3D import InceptionI3d

import neptune.new as neptune

run = neptune.init_run(
	project="wangsonghe1998/asl",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
	source_files=["**/*.py"]
) 


def load_i3d(path):
	state_dict = torch.load(path)['state_dict']
	new_dict = OrderedDict()
	for key, value in state_dict.items():
		key = key.replace('module.','')
		new_dict[key] = value  
	return new_dict


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

prefix = '../../CSL/gloss-zip/color-gloss/resized'
train_batch_size = 8
test_batch_size = 4

train_dataset = Dataset(prefix, mode='train')
test_dataset = Dataset(prefix, mode='test')

train_dataloader = DataLoader(
	train_dataset,
	batch_size=train_batch_size,
	shuffle=True,
	drop_last=True,
	num_workers=0,  # if train_flag else 0
	collate_fn=train_dataset.collate_fn
	)

test_dataloader = DataLoader(
	test_dataset,
	batch_size=test_batch_size,
	shuffle=False,
	drop_last=False,
	num_workers=0,  # if train_flag else 0
	collate_fn=test_dataset.collate_fn
	)


num_data_test = len(test_dataset)
num_data_train = len(train_dataset)
print('number of test data: ', num_data_test)
print('number of train data: ', num_data_train)


num_classes = 500
hidden_size = 512
model = Iso(num_classes, hidden_size)
#model = InceptionI3d(num_classes=1064)
#model.load_state_dict(load_i3d('../../bsl1k_mouth_masked_ppose.pth'))
#model.replace_logits(num_classes)

model.cuda()
pretrain_path = '../work_dir_iso_sd_csl/res+conv_csl_iso_5_0.774.pt'
model.load_state_dict(torch.load(pretrain_path))
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()


# Iterate over data.

epoch = 50

for e in range(epoch):
	accs = []
	for data in train_dataloader:

		padded_video, video_length, label = data
		label = label.cuda()

		try:
			output = model(x=padded_video.cuda(), len_x=video_length.cuda())
		except:
			print(padded_video.shape)
			continue
		loss = loss_fn(output, label)

		prediction = torch.argmax(output, dim=-1)

		acc = torch.sum(label == prediction).cpu() / train_batch_size
		accs.append(acc)
		run['train loss'].log(loss)
		

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	run['train acc'].log(np.mean(accs))

	accs = []
	model.eval()
	for data in test_dataloader:

		padded_video, video_length, label = data
		label = label.cuda()
		try:
			with torch.no_grad():
				output = model(x=padded_video.cuda(), len_x=video_length.cuda())
		except:
			print(padded_video.shape)
			continue

		loss = loss_fn(output, label)

		prediction = torch.argmax(output, dim=-1)
		acc = torch.sum(label == prediction).cpu() / test_batch_size
		accs.append(acc)
		run['test loss'].log(loss)
		

	run['test acc'].log(np.mean(accs))
	rec_acc = round(np.mean(accs),3)
	model_name = f'res+conv_aug_csl_iso_{e}_{rec_acc:.3f}.pt'
	save_path = f'../work_dir_iso_sd_csl/{model_name}'
	torch.save(model.state_dict(), save_path)



# 	loss = torch.mean(anc_feature * pos_feature - anc_feature * neg_feature)
# 	losses.append(loss.item())
# loss_avg = np.mean(losses)
# print(f'{file}, loss:{loss_avg}')









