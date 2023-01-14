"""This script creates main csv files that will work as database for train/val and test.
It receives images folders (patched images, unpatched images) and creates the csv files. 

It receives a folder of images unpatched to get their names, and then (knowing
the patching) creates lists of names ensuring that the splits for train,val and
test sets, as well as the train and val sets of the kfolds contain different images
that are then separated into patches. Ensuring that there is no information of one
set inside anotherone due to the possible overlap when patchifying the images.
Author: Javier Roset"""


import os
import random
import csv

from sklearn.model_selection import KFold

path_images = "D:/UNet_project/images/images_model_best/"
path_masks = "D:/UNet_project/masks/masks_model_best/"
dest_path_csv = "D:/UNet_project/datasets/best_model_patches_overlap_controlled/"
num_folds = 5
y_steps = 5
x_steps = 7

images = os.listdir(path_images)

random.shuffle(images)

testQuant = valQuant = int(len(images)*0.1)

imgsTest = images[:testQuant]
imgsTrainVal = images[testQuant:]

for test in imgsTest:
	if test in imgsTrainVal:
		print(test)


kf = KFold(n_splits = num_folds)

imgsTestAux = []

counter = 0
for image in imgsTest:
	for i in range(y_steps):
		for j in range(x_steps):
			imgsTestAux.append(str(i)+str(j)+image)
			counter += 1 

imgsTrainValAux = []

for image in imgsTrainVal:
	for i in range(y_steps):
		for j in range(x_steps):
			imgsTrainValAux.append(str(i)+str(j)+image) 
			counter += 1

print("counter: " + str(counter))
patches = int(counter/len(images))
print("patches: " + str(patches))

fieldnames = ["files","labels"]

#creating general test file
with open(dest_path_csv + 'test.csv', mode = 'w') as test:
	
	writer = csv.DictWriter(test, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(len(imgsTestAux)):
		writer.writerow({'files':"images/" + imgsTestAux[i],
						'labels':"masks/" + imgsTestAux[i]})
test.close()

#creating general train file
with open(dest_path_csv + 'train.csv', mode = 'w') as train:
	writer = csv.DictWriter(train, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(valQuant*patches,len(imgsTrainValAux)):
		
		writer.writerow({'files':"images/" + imgsTrainValAux[i],
						'labels':"masks/" + imgsTrainValAux[i]})
train.close()

#creating general val file
with open(dest_path_csv + 'val.csv', mode = 'w') as val:

	writer = csv.DictWriter(val, fieldnames = fieldnames)
	writer.writeheader()

	for i in range(valQuant*patches):
		writer.writerow({'files':"images/" + imgsTrainValAux[i],
						'labels':"masks/" + imgsTrainValAux[i]})
val.close()

#creating folds files
fold = 1
for train_index, val_index in kf.split(imgsTrainVal):

	#creationg train file for each fold
	with open(dest_path_csv + 'train_fold_' + str(fold) + ".csv", mode = 'w') as train_fold:

		writer = csv.DictWriter(train_fold, fieldnames = fieldnames)
		writer.writeheader()

		for index in train_index:
			for i in range(y_steps):
				for j in range(x_steps):
					writer.writerow(
						{"files":"images/" + str(i) + str(j) + imgsTrainVal[index],
						"labels":"masks/" + str(i) + str(j) + imgsTrainVal[index]})
	train_fold.close()

	#creating val file for each fold
	with open(dest_path_csv + 'val_fold_' + str(fold) + ".csv", mode = 'w') as val_fold:

		writer = csv.DictWriter(val_fold, fieldnames = fieldnames)
		writer.writeheader()

		for index in val_index:
			for i in range(y_steps):
				for j in range(x_steps):
					writer.writerow(
						{"files":"images/" + str(i) + str(j) + imgsTrainVal[index],
						"labels":"masks/" + str(i) + str(j) + imgsTrainVal[index]})
	val_fold.close()

	fold += 1

