"""This script is only used for the pretraining with uavid database. It can also be used to turn a folder of images to gray."""

import cv2
import os

base = "D:/UNet_project/"
path_or = base + "files_uavid/labeled/images/"
path_dest = base + "files_uavid/labeled/imagesGray/"

images = os.listdir(path_or)

for image_name in images:
	if image_name.split('.')[1] != "png":
		continue
	image = cv2.imread(path_or + image_name)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	cv2.imwrite(path_dest + image_name, gray)