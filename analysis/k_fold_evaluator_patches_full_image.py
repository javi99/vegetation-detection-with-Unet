"""This model makes predictions of a kfold list of models (all 5 models of a kfold) trained over patches reconstructing the full image. It gets a full image of the test set,
brakes it into patches, makes the prediction of the 5 models over a patch, gets the final ponderated prediction of the patch, and then repeats the process untill 
it has all the image, and then it reconstructs the prediction of the full image from all the patches.

The introduction of the files is done by the dataframe method, where a csv file containing all the names of the images for each set is given.
From this file, the images of the test set are selected. A directory containing all the images of the csv file must be given too (path_files)

It also prints in the console the calculated F1, precision and recall for each prediction as well as the global precision, recall and F1 
value at the end of the execution. For evaluation of the model, the images used must be of the test set, or any set of images that the 
model has not seen or compared to while training (nor training or validation set) to ensure that the generalization of the model is being analyzed. 
If not, the results given by this script can be misleading.

It can only be used over full images of the resolution of the camera used in the drones. It cannot be used over patches as it creates the patches
from the images introduced."""

import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append("d:/UNet_project")
from evaluation_functions import *
from matplotlib import pyplot as plt
from patchify import unpatchify
import pandas as pd


base = "D:/UNet_project/"
path_files = base + "files_patches_overlap/"
path_csv = base + "datasets/patches_overlap_controlled/test.csv"
path_weights = base + "final_weights/kfold_overlap_controlled_dataAug_dropout/"
weights_name = "patches_overlap_dropout_dataAug_best_val_prec.hdf5"
path_full_masks = base + "/masks/masksNoTrees/"
path_full_image = base + "/images/imagesGrayNoTrees/"
path_results = base + "results/overlap_controlled_dataAug_dropout/best_images/"

folds = 1
size = 256
full_image_size = (int(2048/2),int(1536/2))
final_size = (int(1536/2),int(2048/2))
seed = 42
threshold = 0.6
threshold_mean_predictions = 1

models = []
predictions = []
precision_l = []
recall_l = []
F1_l = []

#loading all models
for i in range(2, folds + 2):
	models.append(tf.keras.models.load_model(
				path_weights + "fold_" + str(i) + 
				weights_name, compile=False))

csv_file = pd.read_csv(path_csv, dtype = str)
images = []
for index,row in csv_file.iterrows():
	image = row['files'].split('/')[1][2:]
	
	if image not in images:
		images.append(image)

counter = 0
for image_name in images:
	patches = np.zeros((5,7,size,size))
	patches_mask = np.zeros((5,7,size,size))
	for i in range(7):
		for j in range(5):
			path_to_image = os.path.join(path_files,"images/"+str(j)+str(i)+image_name)
			path_to_mask = os.path.join(path_files,"masks/"+str(j)+str(i)+image_name)
			image = cv2.imread(path_to_image)[:,:,0]
			image = cv2.resize(image,(size,size))
			
			for model in models:

				predictions.append(calc_prediction(image, model, threshold))
		
			prediction = np.sum(predictions, axis = 0) >= threshold_mean_predictions
			predictions.clear()

			patches[j,i,:,:] = prediction

	assert patches.shape == (5,7,256,256)
	
	reconstructed_prediction = unpatchify(patches,final_size).astype(bool)

	mask = cv2.imread(path_full_masks + image_name)[:,:,0]
	reconstructed_mask = cv2.resize(mask, full_image_size)
	
	image = cv2.imread(path_full_image + image_name)
	image = cv2.resize(image, full_image_size)

	fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
	axs[0].imshow(image, cmap='gray')
	axs[0].set_title('image')
	axs[1].imshow(reconstructed_mask, cmap = 'gray')
	axs[1].set_title('label')
	axs[2].imshow(reconstructed_prediction, cmap = 'gray')
	axs[2].set_title('prediction')
	fig.tight_layout()
	#plt.show()
	fig.savefig(path_results + image_name)
	plt.close(fig)
    
	precision, recall, F1 = PrecRecF1(reconstructed_mask.astype(bool), reconstructed_prediction)
	print(image_name)
	print("pr: " + str(precision) + " rec: " + str(recall) + " F1: " + str(F1))
	if precision == 0 and recall == 0:
		continue
	precision_l.append(precision)
	recall_l.append(recall)
	F1_l.append(F1)

print(np.mean(precision_l))
print(np.mean(recall_l))
print(np.mean(F1_l))