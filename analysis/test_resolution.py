"""This script grabs all the masks, resizes them to 256,256 and then back to full resolution, and then calculates the F1 score, precision and recall of the masks over
themselves to see the differences produced by the reshaping and the problem it might be when predicting and evaluating the model as it has been impossible to obtain F1 scores 
higher than 60% aprox. When executed, the F1 is approximately of 60%, which gives an idea of the information lost when resizing, and indicates that achieving higher 
performance can be very difficoult, if resizing to 256,256. It would be interesting to plot a graphic showing the F1 of the masks over themselves for several sizes, 
to see if there is another image size interesting for improving the capability of the detecting system, while not surpasing the requirements."""

import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append("E:/UNet_project")
from evaluation_functions import *
from matplotlib import pyplot as plt
from patchify import unpatchify
import pandas as pd

base = "E:/UNet_project/"
path_files = base + "files_patches_overlap/"
path_csv = base + "datasets/all_images/test.csv"

path_full_masks = base + "/masks/masksBinary/"


csv_file = pd.read_csv(path_csv, dtype = str)
images = []
for index,row in csv_file.iterrows():
	image = row['files'].split('/')[1]
	
	if image not in images:
		images.append(image)

precision_l = []
recall_l = []
F1_l = []

for image_name in images:
    mask = cv2.imread(path_full_masks + image_name)[:,:,0]
    print(path_full_masks + image_name)
    size = mask.shape
    print(size)
    resized_mask = cv2.resize(mask,(256,256))
    full_mask = cv2.resize(resized_mask,(size[1],size[0]))
    print(full_mask.shape)

    precision, recall, F1 = PrecRecF1(full_mask, mask)
    print(image_name)
    print("pr: " + str(precision) + " rec: " + str(recall) + " F1: " + str(F1))
    
    precision_l.append(precision)
    recall_l.append(recall)
    F1_l.append(F1)

    cv2.imshow("mascara",mask)
    cv2.imshow("mascara resized",full_mask)
    cv2.waitKey(0)

print(np.mean(precision_l))
print(np.mean(recall_l))
print(np.mean(F1_l))