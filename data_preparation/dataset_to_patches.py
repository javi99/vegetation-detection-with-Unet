import cv2
from patchify import patchify, unpatchify
from PIL import Image
import numpy as np
import os

path = "D:\\UNet_project\\datasets\\dataGrayNoTrees"

dest = "D:\\UNet_project\\datasets\\dataGrayNoTreesPatchesNoOverlap"

routes = ["test_images\\test","test_masks\\test",
		  "train_images\\train","train_masks\\train",
		  "val_images\\val","val_masks\\val"]

size = (2048,1536)

for route in routes:

	route_to_dataset = os.path.join(path,route)
	images = os.listdir(route_to_dataset)

	for image in images:

		if image.split(".")[1] != "png":
			continue

		route_to_image = os.path.join(route_to_dataset,image)
		
		file = cv2.imread(route_to_image)

		file = cv2.resize(file, size, cv2.INTER_LINEAR)

		patches = patchify(file[:,:,0], (512, 512), step=512)
