"""This script is a test trying to detect vegetation in an RGB image turned to grayscale, or only grabbing one of the channels.
The intention of this script is to see if a middle step can be done if the project needs to start working with RGB images. Maybe a trained model 
can be used to accelerate the process of labeling RGB images introducing the model in the cvat program."""

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
path_csv = base + "datasets/all_images/train.csv"
path_weights = base + "final_weights/kfold_final/"
weights_name = "final_dataAug_big_best_val_prec.hdf5"
images_path = base + "/images/imagenesRGB/"
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




#loading all models fold_1patches_overlap_controlled_dataAugen
for i in range(1, folds + 1):
	models.append(tf.keras.models.load_model(
				path_weights + "fold_" + str(i) + 
				weights_name, compile=False))


images = os.listdir(images_path)

#calculating metrics for every image
i = 0
for image_name in images:
	image = cv2.imread(images_path + image_name)
	image = cv2.resize(image,(size,size))

	gray = image[:,:,0]
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print(gray.shape)

	for model in models:
		predictions.append(calc_prediction(gray, model, threshold))
		
	prediction = np.sum(predictions, axis = 0) >= threshold_mean_predictions
	predictions.clear()

	if True:

		"""
		--> plots to make sure the results are coherent
		"""
		pred1 = calc_prediction(gray, models[0], threshold)
		pred2 = calc_prediction(gray, models[1], threshold)
		pred3 = calc_prediction(gray, models[2], threshold)
		pred4 = calc_prediction(gray, models[3], threshold)
		pred5 = calc_prediction(gray, models[4], threshold)
		figure = plt.figure(figsize=(16, 8))
		plt.subplot(331)
		plt.title('image')
		plt.imshow(image, cmap = 'gray')
		plt.subplot(332)
		plt.title('labels')
		plt.imshow(gray, cmap = 'gray')
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
	
	"""precision, recall, F1 = PrecRecF1(mask[:,:,0].astype(bool), prediction)
	print("pr: " + str(precision) + " rec: " + str(recall) + " F1: " + str(F1))
	if precision == 0 and recall == 0:
		continue
	precision_l.append(precision)
	recall_l.append(recall)
	F1_l.append(F1)"""
	
	i += 1

"""print(np.mean(precision_l))
print(np.mean(recall_l))
print(np.mean(F1_l))
"""
