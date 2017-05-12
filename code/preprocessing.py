from __future__ import print_function
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
from skimage import io, transform

PATH_TO_DATA = '../data'

train = pd.read_csv(PATH_TO_DATA + '/driver_imgs_list.csv')

train['id'] = range(train.shape[0])
subj = np.array(train['subject'])
num_subj = np.unique(train['subject'])
imgs = []

for index, row in train.iterrows():
	img_path = PATH_TO_DATA + '/train/'+ str(row[1]) +'/'+ str(row[2])
	img = cv2.imread(img_path)
	img = cv2.resize(img, (160, 120))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	imgs.append(img)

	if row['id'] % 100 == 0:
		print('{} done'.format(row['id']))

imgs = np.array(imgs)
print(imgs.shape)
