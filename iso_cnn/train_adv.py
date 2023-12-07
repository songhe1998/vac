import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_adv import Dataset

from iso_model import Iso

import neptune.new as neptune

run = neptune.init_run(
	project="wangsonghe1998/asl",
	api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZTI3Yjg1Yy05YjJmLTQwNzctOTQxNi00ZjI5ZTY0MWUyMWMifQ==",
	source_files=["**/*.py"]
) 

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
sm = False

gloss_dict = make_gloss_dict([train_gt_path, test_gt_path])
print('number of glosses: ', len(gloss_dict))
for attack in []:
	train_dataset = Dataset(prefix, train_gt_path, gloss_dict, mode='train')
	test_dataset = Dataset(prefix, test_gt_path, gloss_dict, mode='val')

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,  # if train_flag else 0
		collate_fn=train_dataset.collate_fn
		)

	test_dataloader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=0,  # if train_flag else 0
		collate_fn=test_dataset.collate_fn
		)


	num_data_test = len(test_dataset)
	num_data_train = len(train_dataset)
	print('number of test data: ', num_data_test)
	print('number of train data: ', num_data_train)


	num_classes = len(gloss_dict)
	hidden_size = 512
	model = Iso(num_classes, hidden_size)
	model.cuda()
	model.train()
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
	loss_fn = torch.nn.CrossEntropyLoss()


	# Iterate over data.

	epoch = 10

	for e in range(epoch):
		real_accs = []
		fake_accs = []
		corrupt_success = []
		for data in train_dataloader:

			padded_video, video_length, label, fake_label = data
			label = label.cuda()
			fake_label = fake_label.cuda()
			output = model(x=padded_video.cuda(), len_x=video_length.cuda())
			loss = loss_fn(output, fake_label)

			prediction = torch.argmax(output, dim=-1)

			real_acc = torch.sum(label == prediction).cpu() / batch_size
			real_accs.append(real_acc)
			fake_acc = torch.sum(fake_label == prediction).cpu() / batch_size
			fake_accs.append(fake_acc)

			corrupt_index = label != fake_label
			if torch.sum(corrupt_index) != 0:
				cs = torch.sum((fake_label == prediction)[corrupt_index]).cpu() / torch.sum(corrupt_index).cpu()
				corrupt_success.append(cs)
			run[f'{attck} train loss'].log(loss)
			

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		run[f'{attck} train real acc'].log(np.mean(real_accs))
		run[f'{attck} train fake acc'].log(np.mean(fake_accs))
		run[f'{attck} train corrupt acc'].log(np.mean(corrupt_success))

		model.eval()
		real_accs = []
		fake_accs = []
		corrupt_success = []
		for data in test_dataloader:

			padded_video, video_length, label, fake_label = data
			label = label.cuda()
			fake_label = fake_label.cuda()
			with torch.no_grad():
				output = model(x=padded_video.cuda(), len_x=video_length.cuda())

				loss = loss_fn(output, fake_label)

			prediction = torch.argmax(output, dim=-1)

			real_acc = torch.sum(label == prediction).cpu() / batch_size
			real_accs.append(real_acc)
			fake_acc = torch.sum(fake_label == prediction).cpu() / batch_size
			fake_accs.append(fake_acc)

			corrupt_index = label != fake_label
			if torch.sum(corrupt_index) != 0:
				cs = torch.sum((fake_label == prediction)[corrupt_index]).cpu() / torch.sum(corrupt_index).cpu()
				corrupt_success.append(cs)

			run[f'test loss'].log(loss)
			
			

		run[f'{attck} test real acc'].log(np.mean(real_accs))
		run[f'{attck} test fake acc'].log(np.mean(fake_accs))
		run[f'{attck} test corrupt acc'].log(np.mean(corrupt_success))

		rec_acc = round(np.mean(fake_accs),3)
		model_name = f'{attck}_cnn_gsl_iso_{e}_{rec_acc}.pt'
		save_path = f'../work_dir_iso_sd_adv/{model_name}'
		torch.save(model.state_dict(), save_path)



# 	loss = torch.mean(anc_feature * pos_feature - anc_feature * neg_feature)
# 	losses.append(loss.item())
# loss_avg = np.mean(losses)
# print(f'{file}, loss:{loss_avg}')









