"""This script creates main csv files that will work as database for train/val and test.
It receives images folders (unpatched images) and creates the csv files. 

No patches
Author: Javier Roset"""


import os
import random
import csv

from sklearn.model_selection import KFold

base = "E:/UNet_project/"

path_images = base + "images/imagesGray/"
path_masks = base + "masks/masksRaw/"
dest_path_csv = base + "datasets/all_images2/"

num_folds = 5
file_extension = ".png"

images = os.listdir(path_images)

random.shuffle(images)

testQuant = valQuant = int(len(images)*0.1)

imgsTest = images[:testQuant]
imgsTrainVal = images[testQuant:]

kf = KFold(n_splits = num_folds)

imgsTestAux = []
counter = 0
for image in imgsTest:
	
	imgsTestAux.append(image)
	counter += 1 

imgsTrainValAux = []
for image in imgsTrainVal:
	
	imgsTrainValAux.append(image) 
	counter += 1

print("counter: " + str(counter))

fieldnames = ["files","labels"]

#creating general test file
with open(dest_path_csv + 'test.csv', mode = 'w') as test:
	
	writer = csv.DictWriter(test, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(len(imgsTestAux)):
		writer.writerow({'files':"images/" + imgsTestAux[i],
						'labels':"masks/" + imgsTestAux[i].split(".")[0] + file_extension})
test.close()

#creating general train file
with open(dest_path_csv + 'train.csv', mode = 'w') as train:
	writer = csv.DictWriter(train, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(valQuant,len(imgsTrainValAux)):
		
		writer.writerow({'files':"images/" + imgsTrainValAux[i],
						'labels':"masks/" + imgsTrainValAux[i].split(".")[0] + file_extension})
train.close()

#creating general val file
with open(dest_path_csv + 'val.csv', mode = 'w') as val:

	writer = csv.DictWriter(val, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(valQuant):
		writer.writerow({'files':"images/" + imgsTrainValAux[i],
						'labels':"masks/" + imgsTrainValAux[i].split(".")[0] + file_extension})
val.close()

#creating folds files
fold = 1
for train_index, val_index in kf.split(imgsTrainVal):

	#creationg train file for each fold
	with open(dest_path_csv + 'train_fold_' + str(fold) + ".csv", mode = 'w') as train_fold:

		writer = csv.DictWriter(train_fold, fieldnames = fieldnames)
		writer.writeheader()

		for index in train_index:

			writer.writerow(
				{"files":"images/" + imgsTrainVal[index],
				"labels":"masks/" + imgsTrainVal[index].split(".")[0] + file_extension})
	train_fold.close()

	#creating val file for each fold
	with open(dest_path_csv + 'val_fold_' + str(fold) + ".csv", mode = 'w') as val_fold:

		writer = csv.DictWriter(val_fold, fieldnames = fieldnames)
		writer.writeheader()

		for index in val_index:

			writer.writerow(
				{"files":"images/" + imgsTrainVal[index],
				"labels":"masks/" + imgsTrainVal[index].split(".")[0] + file_extension})
	val_fold.close()

	fold += 1

