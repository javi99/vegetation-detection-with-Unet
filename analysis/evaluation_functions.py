"""This script contains all necessary functions for evaluation of the model"""


import numpy as np
import cv2
import tensorflow as tf
import imutils

def calculate_TP_FN_FP(y_true, y_pred):
    """
    This function calculates true positives, false negatives, and false positives
    for latter calculation of precision and recall. 
    
    Inputs:
      - y_true: boolean array of size (size,size) representing the label
      - y_pred: boolean array of size (size,size) representing the output of the model
    
    Outputs:
      - TP: integer representing true positives
      - FP: integer representing false positives
      - FN: integer representing false negatives
    """
    TP = np.sum(np.logical_and(y_true, y_pred))
    FN = np.sum(np.logical_and(y_true, (~y_pred)))
    FP = np.sum(np.logical_and((~y_true), y_pred))

    return TP,FN,FP

def PrecRecF1(y_true, y_pred):
    """
    This function calculates precision, recall and F1 Score. 
    
    Inputs:
      - y_true: boolean array of size (size,size) representing the label
      - y_pred: boolean array of size (size,size) representing the output of the model
    
    Outputs:
      - Precision: float value representing precision
      - Recall: float value representing recall
      - F1 Score: float value representing F1 Score
    """
    TP,FN,FP = calculate_TP_FN_FP(y_true, y_pred)

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1

def calc_prediction(image, model, threshold):

    """
    This function calculates the prediction of a model over an image. 
    Advise: Make sure that the image is of the size that the model accepts.
    Inputs:
      - image: array
      - model: tensorflow model object
      - threshold: float. it indicates the threshold of the prediction

    It returns the prediction as a boolean array of the corresponded size.
    """
    
    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image), axis=1),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    prediction = (model.predict(image_input)[0,:,:,0] > threshold).astype(np.uint8)
    
    return prediction

def calc_prediction_softmax(image, model):

    """
    This function calculates the prediction of a model over an image for a model that classifies different objects (cars, vegetation, buildings...). 
    Advise: Make sure that the image is of the size that the model accepts.
    Inputs:
      - image: array
      - model: tensorflow model object

    It returns the prediction as an integer array indicating the class for each pixel.
    """
    
    image_norm = image[:,:,None]
    image_input = np.expand_dims(image_norm,0)

    prediction = (model.predict(image_input))
    prediction = np.squeeze(prediction, axis = 0)

    prediction_layers = np.zeros(np.shape(prediction))
    prediction_layers[prediction == prediction.max(axis = 2)] = 1

    prediction = np.argmax(prediction, axis=2, out=None)
    
    return prediction,prediction_layers


#_____________________size analysis_________________
"""Size analysis scripts do not work and need revision"""
def getCountours(image):

    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours

def getDetectionsParamsFromMasks(quantities_masks,mask,sizes):

	cnts = getCountours(mask.copy())
	print("len contours"+ str(len(cnts)))
	centroids = np.empty([3,len(cnts)])

	i = 0
	for c in cnts:

		# compute the center of the contour
		M = cv2.moments(c)

		if M["m00"] == 0:
			continue

		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		area = cv2.contourArea(c)

		centroids[0][i] = cX
		centroids[1][i] = cY
		centroids[2][i] = area

		for z in range(0, len(sizes)-1):
			if area > sizes[z] and area < sizes[z+1]:
				quantities_masks[z] = quantities_masks[z] + 1

		i += 1
	print(np.max(centroids[2]))
	return quantities_masks,centroids