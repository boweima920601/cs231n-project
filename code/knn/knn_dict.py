import cv2
import os
from os import listdir, makedirs
from os.path import isfile, join
import numpy as np
import sys
import json
import time

import heapq
import operator

from sklearn.neighbors import NearestNeighbors

# RESULT_PATH = '../data/test_tiny'
RESULT_PATH = 'test_tiny'
if not os.path.exists(RESULT_PATH):
	os.makedirs(RESULT_PATH)

img_cols = 40
img_rows = 30
top_num = 11

onlyfiles = [f for f in listdir(RESULT_PATH) if isfile(join(RESULT_PATH, f)) and f[-3:] == 'jpg']

N = len(onlyfiles)
print(N)
M = np.zeros((N, img_rows * img_cols * 3))
count = 0
d = {}
for file in onlyfiles:
	img = cv2.imread(join(RESULT_PATH, file))
	img_flat = np.reshape(img, [-1])
	M[count] = img_flat
	d[count] = file
	if count != 0 and count % 10000 == 0:
		print(count)
	count += 1

with open('knn_dict.json', 'w') as f:
	json.dump(d, f)