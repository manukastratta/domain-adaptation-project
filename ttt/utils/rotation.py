import torch
import torch.utils.data
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image


import torchvision.transforms as T
from PIL import Image



# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


class RotateImageFolder(Dataset):
	def __init__(self, traindir, labeldir, train_transform, original=True, rotation=True, rotation_transform=None, return_imgnames=False):
		self.original = original
		self.rotation = rotation
		self.rotation_transform = rotation_transform	
		self.image_dir = traindir
		self.metadata = pd.read_csv(labeldir)
		self.transform = train_transform
		self.return_imgnames = return_imgnames

	def __len__(self):
		return len(self.metadata)
		
	def __getitem__(self, index):
		image_path = os.path.join(self.image_dir, self.metadata.iloc[index]['img_filename'])
		image_input = Image.open(image_path)
		if self.transform: image_input = self.transform(image_input) # image.size(): [3, 224, 224]
		target = int(self.metadata.iloc[index]['Black_Hair'])
		results = []
		if self.original:
			results.append(image_input)
			results.append(target)
			if self.return_imgnames: results.append(self.metadata.iloc[index]['img_filename'])

			# transform = T.ToPILImage()
			# save_img = transform(image_input)
			
			# save_img.save('/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/foo.png')
			# print(target)

		if self.rotation:
			if self.rotation_transform is not None:
				transform = T.ToPILImage()
				image_input = transform(image_input)
				img = self.rotation_transform(image_input)
			target_ssh = np.random.randint(0, 4, 1)[0]
			img_ssh = rotate_single_with_label(img, target_ssh)
			results.append(img_ssh)
			results.append(target_ssh)
		return results

	def switch_mode(self, original, rotation):
		self.original = original
		self.rotation = rotation
