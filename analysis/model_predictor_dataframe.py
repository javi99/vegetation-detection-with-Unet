"""This script makes predictions of a single model and creates a plot containing: image + label + prediction for visual comparison.
Then, it saves these plots in the path_results folder. 

The introduction of the files is done by the dataframe method, where a csv file containing all the names of the images for each set is given.
From this file, the images of the test set are selected. A directory containing all the images of the csv file must be given too (path_files)

Advise: make sure the size of the image introduced to do the prediction has the size that the model can accept.

It also prints in the console the calculated F1, precision and recall for eachprediction as well as the global precision, recall and F1 
value at the end of the execution. For evaluation of the model, the images used must be of the test set, or any set of images that the 
model has not seen or compared to while training (nor training or validation set) to ensure that the generalization of the model is being analyzed. 
If not, the results given by this script can be misleading.

It can be used over full images, or patches."""

import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pandas as pd
import sys
sys.path.append("E:/UNet_project/analysis/")
from evaluation_functions import calc_prediction_softmax, PrecRecF1, calc_prediction



base = "E:/Unet_project/"

path_files = base + "files_no_patches_all_images/"
path_csv = base + "datasets/all_images2/test.csv"
path_full_masks = base + path_files + "masks/"
path_full_image = base + path_files + "images/"
path_weights = base + "final_weights/kfold_final2/"
model_name = "fold_2final_dataAug_big_best_val_prec"
path_results = base + "results/final2/model_2_best/"

size = 256
threshold = 0.25

csv_file = pd.read_csv(path_csv, dtype = str)
images = []
for index,row in csv_file.iterrows():
	image = row['files'].split('/')[1]
	print(image)
	images.append(image)

#loading models

model = tf.keras.models.load_model(path_weights + model_name + ".hdf5", compile=False)

precision_l = []
recall_l = []
F1_l = []

image_counter = 0
for image_name in images:
	print(image_name)
	image = cv2.imread(path_full_image + image_name)[:,:,0]
	image = cv2.resize(image, (size,size))
	image = image * 1/255

	prediction = calc_prediction(image, model, threshold)

	mask_image = cv2.imread(path_full_masks + image_name)[:,:,0]
	mask_image = cv2.resize(mask_image, (size,size))

	prec,rec,F1 = PrecRecF1(mask_image, prediction)
	print("pr: " + str(prec) + " rec: " + str(rec) + " F1: " + str(F1))
	precision_l.append(prec)
	recall_l.append(rec)
	F1_l.append(F1)

	fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(16,10))
	axs[0].imshow(image, cmap='gray')
	axs[0].set_title('image')
	axs[1].imshow(prediction, cmap = 'gray')
	axs[1].set_title('gray prediction')
	axs[2].imshow(mask_image, cmap = 'gray')
	axs[2].set_title('mask')
	
	plt.show()
	fig.savefig(path_results + image_name)
	plt.close(fig)

print(np.mean(precision_l))
print(np.mean(recall_l))
print(np.mean(F1_l))
