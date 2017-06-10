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
for file in onlyfiles:
	img = cv2.imread(join(RESULT_PATH, file))
	img_flat = np.reshape(img, [-1])
	M[count] = img_flat
	if count != 0 and count % 10000 == 0:
		print(count)
	count += 1

print(M)
# with open('img_matrix.txt', 'w') as f:
# 	for i in range(N):
# 		for j in range(img_rows * img_cols * 3 - 1):
# 			f.write(str(M[i][j]) + '\t')
# 		f.write(str(M[i][-1]) + '\n')
# 		if i % 10000 == 0:
# 			print(i)


# raise
start = time.time()

nbrs = NearestNeighbors(n_neighbors=top_num, algorithm='auto').fit(M)
_, indices = nbrs.kneighbors(M)

print('start saving!')

# res_dict = {}
# for i in range(N):
# 	m = M[i]
# 	dist = np.linalg.norm(M - m, axis=1)
# 	k_neigh = list(zip(*heapq.nsmallest(top_num, enumerate(dist), key=operator.itemgetter(1))))[0]
# 	# k_neigh = np.argsort(dist)[:11]
# 	res_dict[i] = list(k_neigh)
# 	print(res_dict[i])
# 	if i == 10:
# 		break
with open('knn.npy', 'wb') as f:
	np.save(f, indices)

print(time.time() - start)
print(indices)


# with open('knn.json', 'w') as f:
# 	json.dump(res_dict, f)
