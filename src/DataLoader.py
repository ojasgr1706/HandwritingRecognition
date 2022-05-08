# import torch
# import torchvision
import os
import cv2
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class HTR_Dataset(Dataset):
	def __init__(self, gt, data_path, charlist_path, transform = None):
		self.imgSize = [128,32]
		self.data_path = data_path
		self.gt = gt
		self.lines = open(self.gt, 'r').readlines()
		self.transform = transforms.Compose([
			preprocess,
			transforms.ToTensor(),
		])
		self.mapping = {}
		self.charlist_path = charlist_path
		self.charlist = open(charlist_path, 'r').readlines()[0].strip()
		count = 1
		self.idx_list = list()

		for letter in self.charlist[1:]:
			self.mapping[letter] = count
			count += 1
		# self.data = []
		# self.label = []

	def __len__(self):
		return (len(self.lines))
		# return count


	def __getitem__(self, idx, transform = True):
		# img = self.data[idx]
		# label = self.label[idx]

		self.idx_list.append(idx)
		line = self.lines[idx]
		words = line.split()
		folders = words[0].split('-')
		img_name = self.data_path + '/' + folders[0] + '/' + folders[0] + '-' + folders[1] + '/' + words[0] + '.png'
		# img_name = self.data_path + '/' + words[0][:3] + '/' + words[0][:8] + '/' + words[0] + ".png"

		img = cv2.imread(img_name , cv2.IMREAD_GRAYSCALE)
		label = words[-1]
		if transform:
			img = self.transform(img)
		return img, label

	def show_image(self, idx, transform = False):
		img, label = self.__getitem__(idx)
		img = img.reshape(self.imgSize)
		plt.imshow(img)
		plt.title(label)
		plt.show()


def preprocess(img):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"

	# there are damaged files in IAM dataset - just use black image instead
	imgSize = [128, 32]
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# increase dataset size by applying random stretches to the images
	# if dataAugmentation:
	# 	stretch = (random.random() - 0.5) # -0.5 .. +0.5
	# 	wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
	# 	img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5

	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	# transpose for TF
	img = cv2.transpose(target)

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img

	return img
