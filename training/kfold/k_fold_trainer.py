"""This script is used to train the models. It can be used in dataframe model or directory model.
The different variables allow for the configuration of the training. All the variables that should be changed are
defined in this script."""

import sys
base = "E:/UNet_project/"

sys.path.append(base)
sys.path.append(base + "training")
from general_functions import *
from training_functions import model_training
import os 
import pandas as pd

seed = 24

#General args
dataset_mode = "dataframe"
augmented = True
batch_size = 16
colormode_images = "grayscale"
colormode_masks = "grayscale" #for binary classification
targetSize = (256,256)
epochs = 1000
initial_learning_rate = 1.5e-3
check_matching_images = False

#paths and weights names
path_weights = base + "final_weights/kfold_final2/"
weights_name = "final_dataAug_big"

#for flow_from_dataframe
path_files = base + "files_no_patches_all_images/"
path_datasets = base + "/datasets/all_images2/"

#for flow from directory
path_train_dir = ""
path_val_dir = ""
path_labels_train_dir = ""
path_labels_val_dir = ""

#Pretraining and finetunning args
pretrained = False
finetuning_value = False
path_pretrained = base + "final_weights/uavid_256_noAugmentation_dropout_best_val_prec"
pretrained_folder = base + "final_weights/kfold_pretraining_uavid_smallWeights/"
pretrained_name = "uavid_256_noAugmentation_dropout_best_val_prec"

number_of_folds = 5

for i in range(1,number_of_folds+1):

	if i != 1:
		check_matching_images = False
	

	if dataset_mode == "dataframe":
		path_train = path_datasets + "train_fold_" + str(i) + ".csv"
		path_val = path_datasets + "val_fold_" + str(i) + ".csv"
		
		csv_file = pd.read_csv(path_train,dtype = str)

		train_images, train_masks, input_shape = dataset_from_dataframe(path_train, path_files,  
							augmented, batch_size, seed, colormode_images, colormode_masks, targetSize)
		
		"""En validacion NO ponemos data augmentation, ya que no aporta y nos impide ver como de bien esta aprendiendo el 
			modelo"""
		val_images, val_masks = dataset_from_dataframe(path_val, path_files, 
							False, batch_size, seed, colormode_images, colormode_masks, targetSize)[:2]
		
		csv_file = pd.read_csv(path_train)
		num_training_imgs = len(csv_file)

	if dataset_mode == "directory":
		train_images, train_masks, input_shape = dataset_from_directory(path_train_dir, path_labels_train_dir, 
							augmented, batch_size, seed, colormode_images, colormode_masks, targetSize)
		val_images, val_masks = dataset_from_directory(path_val_dir, path_labels_val_dir, 
							False, batch_size, seed, colormode_images, colormode_masks, targetSize)[:2]
		num_training_imgs = len(os.listdir(path_train_dir+"/train"))

	if pretrained == True:

		path_pretrained = path_pretrained
	
	else:

		path_pretrained = "empty"
	
	if finetuning_value:
		path_pretrained = pretrained_folder + "fold_" + str(i) + pretrained_name


	model_training(train_images, train_masks, val_images, val_masks, input_shape, num_training_imgs,
                 batch_size, epochs, "fold_" + str(i) + weights_name,path_weights,  initial_learning_rate = initial_learning_rate, 
				 finetuning = finetuning_value, check_matching_images = check_matching_images,pretrained = path_pretrained)

	print("training of fold" + str(i) + " done")
	
