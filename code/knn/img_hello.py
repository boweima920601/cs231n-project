import cv2
import os
from os import listdir, makedirs
from os.path import isfile, join
import json


TEST_PATH = '../data/test'
RESULT_PATH = '../data/test_tiny'
if not os.path.exists(RESULT_PATH):
	os.makedirs(RESULT_PATH)
img_cols = 40
img_rows = 30

onlyfiles = [f for f in listdir(TEST_PATH) if isfile(join(TEST_PATH, f))]
l = [0, 3341, 6298, 8805, 5946, 7336, 6824, 8166, 3273, 9622, 7305]
l = [1, 7932, 5137, 3620, 7952, 6173, 2683, 8015, 9832, 9410, 539]
l = [    0, 61692, 69953, 18893,  9333, 53632, 34879,  2899, 70105,
       35470, 33632]
with open('knn_dict.json', 'r') as f:
	d = json.load(f)

for item in l:
	print d[str(item)]