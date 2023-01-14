
"""This script trains 1 model with the directory method. It has been obtained from
Dr. Sreenivas Bhattiprolu. Watch the video for more information."""
# https://youtu.be/csFGTLT6_WQ
"""
Author: Dr. Sreenivas Bhattiprolu
Training and testing for semantic segmentation (Unet) of veg
Uses standard Unet framework with no tricks!
Dataset info: Electron microscopy (EM) dataset from
https://www.epfl.ch/labs/cvlab/data/data-em/
Patches of 256x256 from images and labels 
have been extracted (via separate program) and saved to disk. 
This code uses 256x256 images/masks.
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import sys
sys.path.append("d:/UNet_project")
from general_functions import dataset_from_directory
from training_functions import dice_metric
from unet_model_with_functions_of_blocks import build_unet
from focal_loss import BinaryFocalLoss

seed=24
size = 256
batch_size = 2
augmented = False
model_name = "modelox.hdf5"

path_train_images = "dataGray/train_images/"
path_train_masks = "dataGray/train_masks/"
path_val_images = "dataGray/val_images/"
path_val_masks = "dataGray/val_masks/"
path_test_images = "dataGray/test_images/"
path_test_masks = "dataGray/test_masks/"

num_train_imgs = len(os.listdir(path_train_images+"train/"))
steps_per_epoch = num_train_imgs //batch_size

train_dataset, input_shape = dataset_from_directory(
            path_train_images, path_train_masks, augmented, batch_size,
            seed, "grayscale", (size,size))
val_dataset = dataset_from_directory(
            path_val_images, path_val_masks, augmented, batch_size,
            seed, "grayscale", (size,size))[0]

i = 0
for image, mask in train_dataset:
    
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()
    i+=1
    if i ==3:
        break

model = build_unet(input_shape)



model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2), 
              metrics=[dice_metric])

model.summary()

history = model.fit_generator(train_dataset, validation_data=val_dataset, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=300)

model.save(model_name)

############################################################
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_metric']
#acc = history.history['accuracy']
val_acc = history.history['val_dice_metric']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

#######################################################################


model = tf.keras.models.load_model(model_name, compile=False)
test_dataset = dataset_from_directory(
            path_test_images, path_test_masks, augmented, 
            len(os.listdir(path_test_images+"/images")),
            seed, "grayscale", (size,size))[0]

### Testing on a few test images

i = 0
for image, mask in test_dataset:
    test_img_input=np.expand_dims(image, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(mask[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.show()
    i+=1
    if i == 3:
        break

#IoU for a single image
from tensorflow.keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(mask[:,:,0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())


#Calculate IoU and average
 
import pandas as pd

IoU_values = []
for image, mask in test_dataset:

    image_input=np.expand_dims(image, 0)
    prediction = (model.predict(image_input)[0,:,:,0] > 0.6).astype(np.uint8)
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(mask[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)
    

df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)    