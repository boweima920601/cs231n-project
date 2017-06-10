import cv2
import os
from os import listdir, makedirs
from os.path import isfile, join


TEST_PATH = '../data/test'
RESULT_PATH = '../data/test_tiny'
if not os.path.exists(RESULT_PATH):
	os.makedirs(RESULT_PATH)
img_cols = 40
img_rows = 30

onlyfiles = [f for f in listdir(TEST_PATH) if isfile(join(TEST_PATH, f))]
print len(onlyfiles)
for file in onlyfiles:
	img = cv2.imread(join(TEST_PATH, file))
	resized = cv2.resize(img, (img_cols, img_rows))
	cv2.imwrite(join(RESULT_PATH, file), resized)
