"""This script is only used for the uavid dataset used for pretraining. It can be found in the kaggle web page."""

from PIL import Image, ImageSequence
import numpy as np
from numpy.core.defchararray import index
import matplotlib.pyplot as plt

path1 = "D:/UNet_project/uavid_extracted/labeled/labels/0.png"
path2 = "D:/UNet_project/uavid/uavid_val/seq16/TrainId/000000.png"

image = Image.open(path1)
image = np.array(image)

clr_tab = {}
clr_tab['Clutter'] = [0, 0, 0]
clr_tab['Building'] = [128, 0, 0]
clr_tab['Road'] = [128, 64, 128]
clr_tab['Static_Car'] = [192, 0, 192]
clr_tab['Tree'] = [0, 128, 0]
clr_tab['Vegetation'] = [128, 128, 0]
clr_tab['Human'] = [64, 64, 0]
clr_tab['Moving_Car'] = [64, 0, 128]
    
label_img = np.zeros(shape=(image.shape[0], image.shape[1],3),dtype=np.uint8)
values = list(clr_tab.values())

for tid,val in enumerate(values):
    mask = (image == tid)
    label_img[mask] = val

plt.imshow(label_img)
plt.show()
