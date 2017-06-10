from __future__ import print_function
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
from skimage import io, transform
import os
import glob


# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_cv2(path, img_cols, img_rows, color_type=1):
	# Load as grayscale
	if color_type == 1:
		img = cv2.imread(path, 0)
	elif color_type == 3:
		img = cv2.imread(path)
	# Reduce size
	resized = cv2.resize(img, (img_cols, img_rows))
	return resized


def get_im_cv2_mod(path, img_cols, img_rows, color_type = 1):
	# Load as grayscale
	if color_type == 1:
		img = cv2.imread(path, 0)
	else:
		img = cv2.imread(path)
	# Reduce size
	rotate = random.uniform(-10, 10)
	M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
	img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
	resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
	return resized

def load_train_data(PATH_TO_DATA, cols, rows, color, sample = 0):

	print("Load train images")
	start_time = time.time()
	train = pd.read_csv(PATH_TO_DATA + '/driver_imgs_list.csv')

	num_train = train.shape[0]
	train['id'] = range(num_train)

	num_subj = np.unique(train['subject'])
	imgs = []
	labels = []
	
	if sample == 0:
		for index, row in train.iterrows():
			img_path = PATH_TO_DATA + '/imgs/train/'+ str(row[1]) +'/'+ str(row[2])
			img = get_im_cv2(img_path, cols, rows, color)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			imgs.append(img)

			labels.append(int(list(str(row[1]))[-1]))
			if (row['id'] * 10) % num_train == 0:
				print('Load images {} done out of total {} images'.format(row['id'], num_train))
		subj = np.array(train['subject'])

	else:
		subj = []
		idx = np.random.choice(range(num_train), sample)
		for i in idx:
			row = train.iloc[i]
			img_path = PATH_TO_DATA + '/imgs/train/'+ str(row[1]) +'/'+ str(row[2])
			img = get_im_cv2(img_path, cols, rows, color)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			imgs.append(img)

			labels.append(int(list(str(row[1]))[-1]))   
			subj.append(row['subject'])
		subj = np.array(subj)

	print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
	imgs = np.array(imgs)
	labels = np.array(labels)
	return imgs, labels, subj

def load_test_data(img_rows, img_cols, color_type = 1):
	print('Read test images')
	start_time = time.time()
	path = os.path.join('..', 'input', 'test', '*.jpg')
	files = glob.glob(path)
	X_test = []
	X_test_id = []
	total = 0
	thr = math.floor(len(files)/10)
	for fl in files:
		flbase = os.path.basename(fl)
		img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
		X_test.append(img)
		X_test_id.append(flbase)
		total += 1
		if total%thr == 0:
			print('Read {} images from {}'.format(total, len(files)))
	
	print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
	X_test = np.array(X_test)
	return X_test, X_test_id

if __name__ == "__main__":

	PATH_TO_DATA = '../data'
	cols = 224
	rows = 224
	color = 3
	sample = 100
	train, labels, subj = load_train_data(PATH_TO_DATA, cols, rows, color, sample)
	"""
	test, test_id = load_test_data(cols, rows, 3)
	np.save('test_data', test)
	np.save('test_data_id', test_id)
	"""
	if sample:
	   np.save('train_data' + '_' + str(sample), train)
	   np.save('train_label' + '_' + str(sample), labels)
	   np.save('train_subj' + '_' + str(sample), subj)
	else:
	   np.save('train_data', train)
	   np.save('train_label', labels)
	   np.save('train_subj', subj)        

