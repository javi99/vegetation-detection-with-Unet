"""
This code is used to invert the masks of the forests. When labeling, it is a lot easier to label the empty spaces in the forests
instead of all the vegetation in them. Then, a simple inverting operation of turning the label to black, and the black to label, gives the full label 
of the forest image. 
"""
import cv2
import os
import numpy as np

or_path = "SUDAN/arboles/masks_inverted"
dest_path = "SUDAN/arboles/masks"

images = os.listdir(or_path)

for image in images:

	image_file = cv2.imread(os.path.join(or_path,image))
	
	inverted = np.zeros((image_file.shape))
	inverted[image_file == 0] = 255

	
	cv2.imwrite(os.path.join(dest_path, image), inverted)