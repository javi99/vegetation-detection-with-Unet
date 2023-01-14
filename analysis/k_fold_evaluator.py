"""This model makes predictions of a kfold list of models (all 5 models of a kfold).

The introduction of the files is done by the dataframe method, where a csv file containing all the names of the images for each set is given.
From this file, the images of the test set are selected. A directory containing all the images of the csv file must be given too (path_files)

Advise: make sure the size of the image introduced to do the prediction has the size that the model can accept.

It also prints in the console the calculated F1, precision and recall for each prediction as well as the global precision, recall and F1 
value at the end of the execution. For evaluation of the model, the images used must be of the test set, or any set of images that the 
model has not seen or compared to while training (nor training or validation set) to ensure that the generalization of the model is being analyzed. 
If not, the results given by this script can be misleading.

It can be used over patches or full images."""

import sys
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

sys.path.append("E:/UNet_project")
from general_functions import dataset_from_dataframe
from evaluation_functions import *
from matplotlib import pyplot as plt

base = "E:/UNet_project/"
path_files = base + "files_no_patches_all_images/"
path_csv = base + "datasets/all_images2/test.csv"
path_weights = base + "final_weights/kfold_final2/"
weights_name = "final_dataAug_big_best_val_prec.hdf5"
folds = 5
size = 256
batch_size = len(os.listdir(path_files+"images/"))
seed = 42
#seed = 28
threshold = 0.3
threshold_mean_predictions = 2

models = []
predictions = []
precision_l = []
recall_l = []
F1_l = []


#loading images and masks
img_generator, mask_generator, input_shape = dataset_from_dataframe(path_csv, path_files, False, 
								batch_size,seed,  "grayscale", "grayscale", (size,size))


#loading all models fold_1patches_overlap_controlled_dataAugen
for i in range(1, folds + 1):
	models.append(tf.keras.models.load_model(
				path_weights + "fold_" + str(i) + 
				weights_name, compile=False))

#loading images and masks
img_generator, mask_generator, input_shape = dataset_from_dataframe(path_csv, path_files, False, 
								batch_size,seed,  "grayscale", "grayscale", (size,size))

dataset = zip(img_generator.next(), mask_generator.next())

#calculating metrics for every image
i = 0
for image, mask in dataset:

	for model in models:

		predictions.append(calc_prediction(image, model, threshold))
		
	prediction = np.sum(predictions, axis = 0) >= threshold_mean_predictions
	predictions.clear()

	if True:

		"""
		--> plots to make sure the results are coherent
		"""
		pred1 = calc_prediction(image, models[0], threshold)
		pred2 = calc_prediction(image, models[1], threshold)
		pred3 = calc_prediction(image, models[2], threshold)
		pred4 = calc_prediction(image, models[3], threshold)
		pred5 = calc_prediction(image, models[4], threshold)
		figure = plt.figure(figsize=(16, 8))
		
		plt.subplot(331)
		plt.title('image')
		plt.imshow(image, cmap = 'gray')
		plt.subplot(332)
		plt.title('labels')
		plt.imshow(mask, cmap = 'gray')
		plt.subplot(333)
		plt.title("Final prediction")
		plt.imshow(prediction, cmap = 'gray')
		plt.subplot(334)
		plt.title("Prediction of model 1")
		plt.imshow(pred1, cmap = 'gray')
		plt.subplot(335)
		plt.title("Prediction of model 2")
		plt.imshow(pred2, cmap = 'gray')
		plt.subplot(336)
		plt.title("Prediction of model 3")
		plt.imshow(pred3, cmap = 'gray')
		plt.subplot(337)
		plt.title("Prediction of model 4")
		plt.imshow(pred4, cmap = 'gray')
		plt.subplot(338)
		plt.title("Prediction of model 5")
		plt.imshow(pred5, cmap = 'gray')
		plt.subplots_adjust(left = 0.1,
							bottom = 0.1,
							right=0.9,
							top = 0.9,
							wspace=0.4,
							hspace=0.4)
		figure.tight_layout()
		plt.show()
	
	precision, recall, F1 = PrecRecF1(mask[:,:,0].astype(bool), prediction)
	print("pr: " + str(precision) + " rec: " + str(recall) + " F1: " + str(F1))
	if precision == 0 and recall == 0:
		continue
	precision_l.append(precision)
	recall_l.append(recall)
	F1_l.append(F1)
	
	i += 1

print(np.mean(precision_l))
print(np.mean(recall_l))
print(np.mean(F1_l))

