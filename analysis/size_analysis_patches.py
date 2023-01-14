"""This script does a size analysis, plotting a bar chart indicating the % of objects of each
    size that have not been detected.
    
    CURRENTLY NOT WORKING"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from evaluation_functions import calc_prediction, getCountours, getDetectionsParamsFromMasks
from patchify import unpatchify

base = "D:/Unet_project/"

path_files = base + "files_patches_overlap/"
path_weights = base + "final_weights/kfold_smallWeights_dropout_dataAug/"
path_csv = base + "datasets/patches_overlap_controlled/test.csv"
weights_name = "patches_smallWeights_dropout_dataAug_last.hdf5"
path_full_masks = base + "/masks/masksNoTrees/"
path_full_image = base + "/images/imagesGrayNoTrees/"
path_results = base + "results/smallWeights_dropout_dataAug/"

model = tf.keras.models.load_model(path_weights + "fold_1" + 
				weights_name, compile=False)


size =256
full_image_size = (int(2048/2),int(1536/2))
final_size = (int(1536/2),int(2048/2))
seed=42
threshold = 0.3

sizes = np.linspace(0,240,17)
quantities_masks = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0]
quantities_model = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0]

csv_file = pd.read_csv(path_csv, dtype = str)
images = []
for index,row in csv_file.iterrows():
	image = row['files'].split('/')[1][2:]
	
	if image not in images:
		images.append(image)

processed_images = 0
num = 0
for image_name in images:
	patches = np.zeros((5,7,size,size))
	patches_mask = np.zeros((5,7,size,size))
	for i in range(7):
		for j in range(5):
			path_to_image = os.path.join(path_files,"images/"+str(j)+str(i)+image_name)
			path_to_mask = os.path.join(path_files,"masks/"+str(j)+str(i)+image_name)
			
			image = cv2.imread(path_to_image)[:,:,0]
			image = cv2.resize(image,(size,size))
			
			prediction = calc_prediction(image, model, threshold)

			patches[j,i,:,:] = prediction

	assert patches.shape == (5,7,256,256)
	
	reconstructed_prediction = unpatchify(patches,final_size).astype(bool)

	mask = cv2.imread(path_full_masks + image_name)[:,:,0]
	reconstructed_mask = cv2.resize(mask, full_image_size)
	
	pred_contours = getCountours(reconstructed_prediction.astype(np.uint8))
	quantities_masks, centroids = getDetectionsParamsFromMasks(quantities_masks, mask,sizes)
	print(quantities_masks)
	num += np.sum(np.array(quantities_masks))
	print(reconstructed_prediction)
	cv2.imshow("image",cv2.resize(cv2.imread(path_full_image+image_name),(512,512)))
	cv2.imshow("pred",reconstructed_prediction)
	cv2.waitKey(0)
    #calculates number of missed objects in predictions
	for i in range(0, len(centroids[0])):
        
		point = (centroids[0][i], centroids[1][i])

		pred_detection = 0

		for c in pred_contours:
            
			if cv2.pointPolygonTest(c, point, False) == 1:

				pred_detection = 1

		if pred_detection == 0:

			for z in range(0, len(sizes)-1):
				if centroids[2][i] > sizes[z] and centroids[2][i] < sizes[z+1]:
					quantities_model[z] += 1
    
	processed_images += 1


ymin = 0
ymax = 100

figure = plt.figure(figsize = (16,8))
ax1 = plt.subplot(221)
plt.title("mascaras")
plt.bar(sizes[1:], quantities_masks,3)
ax1.set_ylim(ymin, ymax)
ax2 = plt.subplot(223)
plt.title("model best")
plt.bar(sizes[1:], quantities_model/sum(np.array(quantities_masks))*100,3)
ax2.set_ylim(ymin, ymax)
plt.show()
figure.savefig(path_results + "bar_chart")
plt.close(figure)

print(sizes)
print(quantities_masks)
print(np.sum(np.array(quantities_masks)))
print(quantities_model)
print(np.sum(np.array(quantities_model)))

print(num)
#___________________________________________
"""for image_name in images:
    #ensuring that the file is png type
    if image.split('.')[1] != "png":
        continue

    mask = imread(path_masks + image)
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE,SIZE))
    mask = np.array(mask)

    #getting shapes from masks
    quantities_masks, centroids = getDetectionsParamsFromMasks(quantities_masks, mask)


    #computing predictions
    file = cv2.imread(path_images + image)[:,:,0]
    file = cv2.resize(file,(SIZE,SIZE), cv2.INTER_LINEAR)
    predictions = calc_predictions(file)

    #getting predictions contours
    pred_cnts_best = getCountours(predictions[:,:,0].astype(np.uint8))

    pred_cnts_model11 = getCountours(predictions[:,:,1].astype(np.uint8))

    #calculates number of missed objects in predictions for each model and size
    for i in range(0, len(centroids[0])):
        
        
        point = (centroids[0][i], centroids[1][i])

        #model best
        pred_detection = 0

        for c in pred_cnts_best:
            
            if cv2.pointPolygonTest(c, point, False) == 1:

                pred_detection = 1

        if pred_detection == 0:

            for z in range(0, len(sizes)-1):
                if centroids[2][i] > sizes[z] and centroids[2][i] < sizes[z+1]:
                    quantities_best[z] += 1
        #model 11
        pred_detection = 0

        for c in pred_cnts_model11:

            if cv2.pointPolygonTest(c, point, False) == 1:

                pred_detection = 1

        if pred_detection == 0:

            for z in range(0, len(sizes)-1):
                if centroids[2][i] > sizes[z] and centroids[2][i] < sizes[z+1]:
                    quantities_model11[z] += 1
    
    processed_images += 1
    print(processed_images)



"""

   