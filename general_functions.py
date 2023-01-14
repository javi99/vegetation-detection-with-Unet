"""Functions for loading the data to the network while training. Creator of dataset/dataframe objects for tensorflow.
    The objects allow to set parameters such as data augmentation, batch size, input size for training, and color mode among others."""

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dataset_from_directory(path_images, path_labels, augmented, batch_size, seed,  colormode_images, colormode_masks, targetSize):
    """
    this function returns a zip of images and labels created with the flow_from_directory method from tensorflow.
    As inputs it receives:
      - path_images: str. path to the images folder. Has to be a parent folder. for example:
                  test_images/
                    test/
                      image1
                      image2
                      ...
      - path_labels: str. path to the labels folder
      - augmented: boolean indicating if data augmentation is wanted
      - batch_size: quantity of images per iteration
      - colormode:  str. color mode options: "grayscale", "rgb", "rgba"
      - target_size: tuple. size wanted for the images. 

      example of use: dataset = dataset_from_directory(path/to/dataset, "grayscale", (512,512))

      
    """

    #choosing the modifications done to the image
    if augmented:
        img_data_gen_args = dict(rescale = 1/255.,
                         #rotation_range=90,
                      #width_shift_range=0.3,
                      #height_shift_range=0.3,
                      #shear_range=0.5,
                      #zoom_range=0.3,
                      horizontal_flip=True)
                      #vertical_flip=True,
                      #fill_mode='reflect')

        mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                       # rotation_range=90,
                      #width_shift_range=0.3,
                      #height_shift_range=0.3,
                      #shear_range=0.5,
                      #zoom_range=0.3,
                      horizontal_flip=True,
                      #vertical_flip=True,
                      #fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))
                       

    if not augmented:
        img_data_gen_args = dict(rescale = 1/255.)

        mask_data_gen_args = dict(rescale = 1/255.,
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) 

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    #creating the images reading object
    img_generator = image_data_generator.flow_from_directory(path_images, 
                                                                seed=seed, 
                                                                batch_size=batch_size,
                                                                color_mode = colormode_images, 
                                                                target_size = targetSize,
                                                                class_mode=None) #Default batch size 32, if not specified here
    #creating the masks reading object
    mask_generator = mask_data_generator.flow_from_directory(path_labels, 
                                                                seed=seed, 
                                                                batch_size=batch_size, 
                                                                color_mode = colormode_masks,   #Read masks in grayscale
                                                                target_size = targetSize,
                                                                class_mode=None)  #Default batch size 32, if not specified here

    x = img_generator.next()
    y = mask_generator.next()
    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return img_generator, mask_generator, input_shape


def dataset_from_dataframe(path_to_csv_file, path_files, augmented, batch_size, seed, colormode_images, colormode_masks, targetSize):
    """
  this function returns a dictionary of "images" and "labels" created with the flow_from_dataframe method from tensorflow.
  As inputs it receives:
    - path_to_csv_file: path to the csv file containing all the nanes of the images that are masks/images
    - path_files: path to the folder containing images and masks folders
    - augmented: boolean indicating if data augmentation is wanted
    - batch_size: quantity of images per iteration
    - seed. Parameter to set the random seed. Allows randomisation to keep constant.
    - colormode_images:  str. color mode options: "grayscale", "rgb", "rgba"
    - colormode_masks:  str. color mode options: "grayscale", "rgb", "rgba"
    - target_size: tuple. size wanted for the images. 

    example of use: dataset = dataset_from_directory(path/to/csv, path/to/files, True, 2, 42, "grayscale", "grayscale", (512,512))

    
    """
    csv_file = pd.read_csv(path_to_csv_file,dtype = str)

    if augmented:
        img_data_gen_args = dict(rescale = 1/255.,
                         #rotation_range=90,
                      #width_shift_range=0.3,
                      #height_shift_range=0.3,
                      #shear_range=0.5,
                      #zoom_range=0.3,
                      horizontal_flip=True,
                      #vertical_flip=True,
                      #fill_mode='wrap'
                      )

        mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 and 1
                        #rotation_range=90,
                      #width_shift_range=0.3,
                      #height_shift_range=0.3,
                      #shear_range=0.5,
                      #zoom_range=0.3,
                      horizontal_flip=True,
                      #vertical_flip=True,
                      #fill_mode='wrap',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) 

    if not augmented:
        img_data_gen_args = dict(rescale = 1/255.)

        mask_data_gen_args = dict(rescale = 1/255.,
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) 
  
    img_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

    img_generator = img_data_generator.flow_from_dataframe(
                                                          dataframe = csv_file,
                                                          directory = path_files,
                                                          x_col = "files",
                                                          y_col = "labels",
                                                          subset = None,
                                                          batch_size = batch_size,
                                                          seed = seed,
                                                          shuffle = True,
                                                          class_mode = None,
                                                          target_size = targetSize,
                                                          color_mode = colormode_images)

    mask_generator = mask_data_generator.flow_from_dataframe(
                                                          dataframe = csv_file,
                                                          directory = path_files,
                                                          x_col = "labels",
                                                          y_col = "files",
                                                          subset = None,
                                                          batch_size = batch_size,
                                                          seed = seed,
                                                          shuffle = True,
                                                          class_mode = None,
                                                          target_size = targetSize,
                                                          color_mode = colormode_masks)
    
    x = img_generator.next()
    y = mask_generator.next()
    
    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return img_generator, mask_generator, input_shape