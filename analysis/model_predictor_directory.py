"""This script makes predictions of a single model and creates a plot containing: image + label + prediction for visual comparison.
Then, it saves these plots in the path_results folder. 

The introduction of the files is done by the directory method, where a directory containing the images to evaluate the model is given.

Advise: make sure the size of the image introduced to do the prediction has the size that the model can accept.

It also prints in the console the calculated F1, precision and recall for eachprediction as well as the global precision, recall and F1 
value at the end of the execution. For evaluation of the model, the images used must be of the test set, or any set of images that the 
model has not seen or compared to while training (nor training or validation set) to ensure that the generalization of the model is being analyzed. 
If not, the results given by this script can be misleading.

It can be used over full images, or patches."""

import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append("E:/UNet_project")

from evaluation_functions import *
from matplotlib import pyplot as plt


base = "E:/UNet_project/"
path_images_all = base + "images/imagesGrayPatches/"
path_images_best = base + "images/model_best_patches_images/"
path_weights = base + "final_weights/sprint1/kfold_best_model_dropout_dataAug/"
path_full_masks = "E:/UNet_project/masks/masksPatches/"
path_results = base + "results/best_model_patches_dropout_dataAug/"
size = 256
seed = 42
threshold = 0.15

models_names = ["fold_1patches_overlap_best_dropout_dataAug_best_val_prec.hdf5"]
models = []
predictions = []
precision_l = []
recall_l = []
F1_l = []

#loading all models fold_1patches_overlap_controlled_dataAugen
for model in models_names:
	models.append(tf.keras.models.load_model(
				path_weights + model, compile=False))

images = []
images_all = os.listdir(path_images_all)
images_best = os.listdir(path_images_best)

for image in images_all:
	if image not in images_best:
		images.append(image)


counter = 0
for image_name in images:
	
	path_to_image = os.path.join(path_images_all,image_name)
	path_to_mask = os.path.join(path_full_masks, image_name)
	print(path_to_mask)
	image = cv2.imread(path_to_image)[:,:,0]
	mask = cv2.imread(path_to_mask)[:,:,0]
	image = cv2.resize(image,(size,size))
	mask = cv2.resize(mask, (size,size))
			
	
	prediction = calc_prediction(image, models[0], threshold)
	if counter < 10:
		fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
		axs[0].imshow(image, cmap='gray')
		axs[0].set_title('image')
		axs[1].imshow(mask, cmap = 'gray')
		axs[1].set_title('label')
		axs[2].imshow(prediction, cmap = 'gray')
		axs[2].set_title('prediction')
		fig.tight_layout()
		plt.show()
		#fig.savefig(path_results + image_name)
		plt.close(fig)
		counter += 1

	"""if counter < 3:
		figure = plt.figure(figsize=(16, 8))
		plt.subplot(331)
		plt.title('image')
		plt.imshow(image, cmap = 'gray')
		plt.subplot(332)
		plt.title('label')
		plt.imshow(mask, cmap = 'gray')
		plt.subplot(333)
		plt.title('prediction')
		plt.imshow(prediction, cmap = 'gray')
		plt.show()
		counter += 1"""
    
	precision, recall, F1 = PrecRecF1(mask.astype(bool), prediction)
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