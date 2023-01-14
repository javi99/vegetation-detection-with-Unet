"""This script contains all functions needed for training"""

import os
import focal_loss
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
import tensorflow as tf
import datetime

from keras import backend as K
from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss

import sys
sys.path.append("E:/UNet_project")
from unet_model_with_functions_of_blocks import build_unet_binary_pretraining, build_unet_binary_finetunning,build_unet_binary
import tensorflow_addons as tfa


def check_images(dataset):
    i = 0
    for data in dataset:
        print("image: " + str(np.shape(data[0])))
        print("mask: " + str(np.shape(data[1])))
            
        plt.subplot(2,2,1)
        plt.imshow(data[0][0,:,:,0], cmap='gray')
        plt.subplot(2,2,2)
        plt.imshow(data[1][0,:,:,0], cmap = "gray")
        plt.subplot(2,2,3)
        plt.imshow(data[0][1,:,:,0], cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(data[1][1,:,:,0], cmap='gray')
        plt.show()

        if i == 2:
            break
        i +=1

#Dice metric can be a great metric to track accuracy of semantic segmentation.
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    # if y_pred.sum() == 0 and y_pred.sum() == 0:
    #     return 1.0
    return 2*intersection / union

Precision = metrics.Precision(thresholds = 0.2)
Recall = metrics.Recall(thresholds = 0.2)
TruePositives = metrics.TruePositives(thresholds = 0.2)
TrueNegatives = metrics.TrueNegatives(thresholds = 0.2)
FalsePositives = metrics.FalsePositives(thresholds = 0.2)
FalseNegatives = metrics.FalseNegatives(thresholds = 0.2)

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision
        self.recall_fn = Recall

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)

def model_training(train_images, train_masks, val_images, val_masks, input_shape,
         num_training_imgs,batch_size, epochs, model_name, dest_path, initial_learning_rate = 1e-3,finetuning = False, pretrained = "empty", 
         check_matching_images =  False):
    """This function allows to train a model. It saves it with a determinated name, and returns
    the model. The arguments are:
    - train_images: object ImageDataGenerator; an ImageDataGenerator containing all training images
    - train_masks: object ImageDataGenerator; an ImageDataGenerator containing all training masks
    - val_images: object ImageDataGenerator; an ImageDataGenerator containing all validation images
    - val_masks: object ImageDataGenerator; an ImageDataGenerator containing all validation masks
    - input_shape: list of int; the shape of the images to train the model on
    - num_training_imgs: int; the number of images the training set has
    - batch_size: int; the batch size used for training (number of images to evaluate at each iteration)
    - epochs: int, number of epochs
    - model_name: str; name to identify the trained model
    - dest_path: str; path where the wheights will be saved
    - initial_learning_rate: float; initial learning rate of the model
    - finetuning: boolean; true if finetunning is being made
    - pretrained: str, the name of a preexisting model to be trained further if exists
    """
    #Creation of callbacks
    log_dir = dest_path + "/logs/fit/" + model_name\
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience = 300, verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,
                                                         histogram_freq = 1)

    checkpoint_path_best = os.path.join(dest_path,model_name + "_best_val_F1.hdf5")
    checkpoint_path_last = os.path.join(dest_path,model_name + "_last.hdf5")

    cp_callback_best = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_best,
                                                 save_best_only=True,
                                                 monitor = "val_f1_score",
                                                 mode = "max",
                                                 verbose=1)

    cp_callback_last = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path_last,
                                                        verbose = 0,
                                                        save_freq = 'epoch')

    #definition of callbacks list for the model
    callbacks_list = [cp_callback_best, cp_callback_last, es_callback, tensorboard_callback]

    #definition of metrics list for the model    
    metrics_list = [dice_metric, Precision, Recall, F1_Score(), TruePositives, 
                TrueNegatives,FalsePositives, FalseNegatives]

    #zipping training and validation sets for introduction in the fit_generator
    train_dataset = zip(train_images, train_masks)
    val_dataset = zip(val_images, val_masks)
    
    #Checking of matching in validation or train images
    if check_matching_images == True:
        
        check_images(train_dataset)
        check_images(val_dataset)

    #loading model. Existing model if wanted, or new model. 

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                initial_learning_rate,
                                                decay_steps=epochs,
                                                decay_rate=0.60,
                                                staircase=False)


    if pretrained != "empty":

        #setting different learning rates for different layers
        
        if finetuning:
            model = tf.keras.models.load_model(pretrained + ".hdf5", compile=False)
            model = build_unet_binary_finetunning(input_shape,model)
            optimizer =  Adam(learning_rate
                = lr_schedule)
            
            #optimizers = [tf.keras.optimizers.Adam(learning_rate = 1e-4),
            #            tf.keras.optimizers.Adam(learning_rate = 1e-3)]
            #optimizers_and_layers = [(optimizers[0], model.layers[:24]),(optimizers[1],model.layers[25:])]
            #optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        else:
            model = tf.keras.models.load_model(pretrained + ".hdf5", compile=False)
            
            #model = build_unet_binary_finetunning(input_shape,model)
            model = build_unet_binary_pretraining(input_shape,model)

            
            optimizer =  Adam(learning_rate
                = lr_schedule)

    else:
        
        model = build_unet_binary(input_shape)
        optimizer =  Adam(learning_rate = lr_schedule)
        
    model.compile(optimizer= optimizer, loss=BinaryFocalLoss(gamma = 2), 
                metrics=metrics_list)
        #model = build_effienet_unet(input_shape)

    #compiling model
    # binary classification
    # model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2), 
    #               metrics=metrics_list)
    

    model.summary()
    
    steps_per_epoch = num_training_imgs //batch_size
    
    #fitting model
    history = model.fit_generator(train_dataset, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_data = val_dataset,
                        validation_steps=steps_per_epoch, 
                        epochs=epochs,
                        callbacks = callbacks_list,
                        verbose = 1)

    #plotting loss function
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #saving lss function and trained model
    plt.savefig( os.path.join(dest_path,model_name + ".png"))
    model.save( os.path.join(dest_path, model_name + '_end.hdf5'))
    
    
