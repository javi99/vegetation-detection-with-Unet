"This script turns all images of a folder into patches. It separates an image into smaller parts of itself"
import cv2
from patchify import patchify, unpatchify
from PIL import Image
import numpy as np
import os

size = (2048,1536)

path = "D:/UNet_project/images/images_model_best"
path_masks = "D:/UNet_project/masks/masks_model_best"
dest = "D:/UNet_project/images/model_best_patches"
dest_masks = "D:/UNet_project/masks/model_best_patches"
patch_size = (512,512)
patch_step = 256

images = os.listdir(path)

for image in images:

	if image.split(".")[1] != "png":
		continue

	route_to_image = os.path.join(path,image)
		
	file = cv2.imread(route_to_image)

	file = cv2.resize(file, size, cv2.INTER_LINEAR)

	patches = patchify(file[:,:,0], patch_size, step=patch_step)
	for i in range(patches.shape[0]):
		for j in range(patches.shape[1]):
												
			cv2.imwrite(os.path.join(dest,str(i)+str(j)+image) , patches[i,j,:,:])
			

masks = os.listdir(path_masks)

for mask in masks:

	if mask.split(".")[1] != "png":
		continue

	route_to_mask = os.path.join(path_masks,mask)
		
	file = cv2.imread(route_to_mask)

	file = cv2.resize(file, size, cv2.INTER_LINEAR)

	patches = patchify(file[:,:,0], (512, 512), step=256)
	for i in range(patches.shape[0]):
		for j in range(patches.shape[1]):
												
			cv2.imwrite(os.path.join(dest_masks,str(i)+str(j)+mask) , patches[i,j,:,:])
