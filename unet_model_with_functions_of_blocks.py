"""
This code includes the different architectures used for different cases. Pretraining with uavid database, finetunning, training from 0 for binary classification,
and training from 0 for multiclass classification
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0


def conv_block(input, num_filters,rate):
    x = Conv2D(num_filters, 3, padding="same")(input)# kernel_regularizer=l2(0.0005)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Dropout(rate)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)# kernel_regularizer=l2(0.0005)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters,rate):
    x = conv_block(input, num_filters,rate)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters,rate):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)# kernel_regularizer=l2(0.0005)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters,rate)
    return x

#Build Unet using the blocks
def build_unet_multiclass(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32,0.1)
    s2, p2 = encoder_block(p1, 64,0.1)
    s3, p3 = encoder_block(p2, 128,0.2)
    s4, p4 = encoder_block(p3, 256,0.2)

    b1 = conv_block(p4, 512,0.3) #Bridge

    d1 = decoder_block(b1, s4, 256,0.2)
    d2 = decoder_block(d1, s3, 128,0.2)
    d3 = decoder_block(d2, s2, 64,0.1)
    d4 = decoder_block(d3, s1, 32,0.1)

    #outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)
    outputs = Conv2D(n_classes,(1,1), activation="softmax")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model

def build_unet_binary(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64,0.1)
    s2, p2 = encoder_block(p1, 128,0.1)
    s3, p3 = encoder_block(p2, 256,0.2)
    s4, p4 = encoder_block(p3, 512,0.2)

    b1 = conv_block(p4, 1024,0.3) #Bridge

    d1 = decoder_block(b1, s4, 512,0.2)
    d2 = decoder_block(d1, s3, 256,0.2)
    d3 = decoder_block(d2, s2, 128,0.1)
    d4 = decoder_block(d3, s1, 64,0.1)

    #outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)
    outputs = Conv2D(1,(1,1), activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")

    i = 0

    for layer in model.layers:
        print(str(i) +": " + str(layer.name) + ", trainable: " + str(layer.trainable))
        print(layer.get_output_at(0).get_shape().as_list())
        i += 1
    return model

def build_unet_binary_pretraining(input_shape, model):
    inputs = Input(input_shape)
    encoder = model

    #1st encoder block
    #convolutions
    x1 = encoder.get_layer(index = 1)(inputs)# kernel_regularizer=l2(0.0005)
    x2 = encoder.get_layer(index = 2)(x1)   #Not in the original network. 
    x3 = encoder.get_layer(index = 3)(x2)
    x4 = encoder.get_layer(index = 4)(x3)
    x5 = encoder.get_layer(index = 5)(x4)
    x6 = encoder.get_layer(index = 6)(x5)
    s1 = encoder.get_layer(index = 7)(x6) 
    #max pooling
    p1 = encoder.get_layer(index = 8)(s1)

    #__________________________________

    #2st encoder block
    #convolutions
    x7 = encoder.get_layer(index = 9)(p1)# kernel_regularizer=l2(0.0005)
    x8 = encoder.get_layer(index = 10)(x7)   #Not in the original network. 
    x9 = encoder.get_layer(index = 11)(x8)
    x10 = encoder.get_layer(index = 12)(x9)
    x11 = encoder.get_layer(index = 13)(x10)
    x12 = encoder.get_layer(index = 14)(x11)
    s2 = encoder.get_layer(index = 15)(x12) 
    #max pooling
    p2 = encoder.get_layer(index = 16)(s2)

    #__________________________________

    #3st encoder block
    #convolutions
    x13 = encoder.get_layer(index = 17)(p2)# kernel_regularizer=l2(0.0005)
    x14 = encoder.get_layer(index = 18)(x13)   #Not in the original network. 
    x15 = encoder.get_layer(index = 19)(x14)
    x16 = encoder.get_layer(index = 20)(x15)
    x17 = encoder.get_layer(index = 21)(x16)
    x18 = encoder.get_layer(index = 22)(x17)
    s3 = encoder.get_layer(index = 23)(x18) 
    #max pooling
    p3 = encoder.get_layer(index = 24)(s3)

    #__________________________________

    #4st encoder block
    #convolutions
    x19 = encoder.get_layer(index = 25)(p3)# kernel_regularizer=l2(0.0005)
    x20 = encoder.get_layer(index = 26)(x19)   #Not in the original network. 
    x21 = encoder.get_layer(index = 27)(x20)
    x22 = encoder.get_layer(index = 28)(x21)
    x23 = encoder.get_layer(index = 29)(x22)
    x24 = encoder.get_layer(index = 30)(x23)
    s4 = encoder.get_layer(index = 31)(x24) 

    #max pooling
    p4 = encoder.get_layer(index = 32)(s4)
    #__________________________________
    ##Bridge
    x25 = encoder.get_layer(index =33)(p4)
    x26 = encoder.get_layer(index = 34)(x25)
    x27 = encoder.get_layer(index = 35)(x26)
    x28 = encoder.get_layer(index = 36)(x27)
    x29 = encoder.get_layer(index = 37)(x28)
    x30 = encoder.get_layer(index = 38)(x29)
    b1 = encoder.get_layer(index = 39)(x30)
    
    #_______________________________
    #decoder block 1
    x31 = encoder.get_layer(index = 40)(b1)
    x32 = encoder.get_layer(index = 41)([x31, s4])
    x33 = encoder.get_layer(index = 42)(x32)
    x34 = encoder.get_layer(index = 43)(x33)
    x35 = encoder.get_layer(index = 44)(x34)
    x36 = encoder.get_layer(index = 45)(x35)
    x37 = encoder.get_layer(index = 46)(x36)
    x38 = encoder.get_layer(index = 47)(x37)
    d1 = encoder.get_layer(index = 48)(x38)
    #___________________________
    #decoder block_2
    x39 = encoder.get_layer(index =49)(d1)
    x40 = encoder.get_layer(index = 50)([x39, s3])   
    x41 = encoder.get_layer(index =51)(x40)
    x42 = encoder.get_layer(index =52)(x41)
    x43 = encoder.get_layer(index =53)(x42)
    x44 = encoder.get_layer(index =54)(x43)
    x45 = encoder.get_layer(index =55)(x44)
    x46 = encoder.get_layer(index =56)(x45)
    d2 = encoder.get_layer(index =57)(x46)
    #___________________________
    #decoder block_3
    x47 = encoder.get_layer(index =58)(d2)
    x48 = encoder.get_layer(index = 59)([x47, s2])
    x49 = encoder.get_layer(index =60)(x48)
    x50 = encoder.get_layer(index = 61)(x49)
    x51 = encoder.get_layer(index = 62)(x50)
    x52 = encoder.get_layer(index = 63)(x51)
    x53 = encoder.get_layer(index = 64)(x52)
    x54 = encoder.get_layer(index = 65)(x53)
    d3 = encoder.get_layer(index = 66)(x54)
    #___________________________
    #decoder block_4
    x55 = encoder.get_layer(index = 67)(d3)
    x56 = encoder.get_layer(index = 68)([x55, s1])
    x57 = encoder.get_layer(index = 69)(x56)
    x58 = encoder.get_layer(index = 70)(x57)
    x59 = encoder.get_layer(index =71)(x58)
    x60 = encoder.get_layer(index = 72)(x59)
    x61 = encoder.get_layer(index = 73)(x60)
    x62 = encoder.get_layer(index = 74)(x61)
    d4 = encoder.get_layer(index = 75)(x62)
    #_____________________
    outputs = Conv2D(1,(1,1), activation="sigmoid", name = "conv2d_output")(d4) 
    model = Model(inputs, outputs, name="U-Net")
    #freezing all layers from 0 to 16
    for i in range(0,25):
        model.layers[i].trainable = False
    i = 0
    for layer in model.layers:
        print(str(i) +": " + str(layer.name) + ", trainable: " + str(layer.trainable))
        print(layer.get_output_at(0).get_shape().as_list())
        i += 1
    return model

def build_unet_binary_finetunning(input_shape, model):
    inputs = Input(input_shape)

    encoder = model

    #1st encoder block
    #convolutions
    x1 = encoder.get_layer(index = 1)(inputs)# kernel_regularizer=l2(0.0005)
    x2 = encoder.get_layer(index = 2)(x1)   #Not in the original network. 
    x3 = encoder.get_layer(index = 3)(x2)
    x4 = encoder.get_layer(index = 4)(x3)
    x5 = encoder.get_layer(index = 5)(x4)
    x6 = encoder.get_layer(index = 6)(x5)
    s1 = encoder.get_layer(index = 7)(x6) 
    #max pooling
    p1 = encoder.get_layer(index = 8)(s1)
    #__________________________________
    #2st encoder block
    #convolutions
    x7 = encoder.get_layer(index = 9)(p1)# kernel_regularizer=l2(0.0005)
    x8 = encoder.get_layer(index = 10)(x7)   #Not in the original network. 
    x9 = encoder.get_layer(index = 11)(x8)
    x10 = encoder.get_layer(index = 12)(x9)
    x11 = encoder.get_layer(index = 13)(x10)
    x12 = encoder.get_layer(index = 14)(x11)
    s2 = encoder.get_layer(index = 15)(x12) 
    #max pooling
    p2 = encoder.get_layer(index = 16)(s2)
    #__________________________________
    #3st encoder block
    #convolutions
    x13 = encoder.get_layer(index = 17)(p2)# kernel_regularizer=l2(0.0005)
    x14 = encoder.get_layer(index = 18)(x13)   #Not in the original network. 
    x15 = encoder.get_layer(index = 19)(x14)
    x16 = encoder.get_layer(index = 20)(x15)
    x17 = encoder.get_layer(index = 21)(x16)
    x18 = encoder.get_layer(index = 22)(x17)
    s3 = encoder.get_layer(index = 23)(x18) 
    #max pooling
    p3 = encoder.get_layer(index = 24)(s3)
    #__________________________________
    #4st encoder block
    #convolutions
    x19 = encoder.get_layer(index = 25)(p3)# kernel_regularizer=l2(0.0005)
    x20 = encoder.get_layer(index = 26)(x19)   #Not in the original network. 
    x21 = encoder.get_layer(index = 27)(x20)
    x22 = encoder.get_layer(index = 28)(x21)
    x23 = encoder.get_layer(index = 29)(x22)
    x24 = encoder.get_layer(index = 30)(x23)
    s4 = encoder.get_layer(index = 31)(x24) 
    #max pooling
    p4 = encoder.get_layer(index = 32)(s4)
    #__________________________________
    ##Bridge
    x25 = encoder.get_layer(index =33)(p4)
    x26 = encoder.get_layer(index = 34)(x25)
    x27 = encoder.get_layer(index = 35)(x26)
    x28 = encoder.get_layer(index = 36)(x27)
    x29 = encoder.get_layer(index = 37)(x28)
    x30 = encoder.get_layer(index = 38)(x29)
    b1 = encoder.get_layer(index = 39)(x30)
    
    #_______________________________
    #decoder block 1
    x31 = encoder.get_layer(index = 40)(b1)
    x32 = encoder.get_layer(index = 41)([x31, s4])
    x33 = encoder.get_layer(index = 42)(x32)
    x34 = encoder.get_layer(index = 43)(x33)
    x35 = encoder.get_layer(index = 44)(x34)
    x36 = encoder.get_layer(index = 45)(x35)
    x37 = encoder.get_layer(index = 46)(x36)
    x38 = encoder.get_layer(index = 47)(x37)
    d1 = encoder.get_layer(index = 48)(x38)
    #___________________________
    #decoder block_2
    x39 = encoder.get_layer(index =49)(d1)
    x40 = encoder.get_layer(index = 50)([x39, s3])   
    x41 = encoder.get_layer(index =51)(x40)
    x42 = encoder.get_layer(index =52)(x41)
    x43 = encoder.get_layer(index =53)(x42)
    x44 = encoder.get_layer(index =54)(x43)
    x45 = encoder.get_layer(index =55)(x44)
    x46 = encoder.get_layer(index =56)(x45)
    d2 = encoder.get_layer(index =57)(x46)
    #___________________________
    #decoder block_3
    x47 = encoder.get_layer(index =58)(d2)
    x48 = encoder.get_layer(index = 59)([x47, s2])
    x49 = encoder.get_layer(index =60)(x48)
    x50 = encoder.get_layer(index = 61)(x49)
    x51 = encoder.get_layer(index = 62)(x50)
    x52 = encoder.get_layer(index = 63)(x51)
    x53 = encoder.get_layer(index = 64)(x52)
    x54 = encoder.get_layer(index = 65)(x53)
    d3 = encoder.get_layer(index = 66)(x54)
    #___________________________
    #decoder block_4
    x55 = encoder.get_layer(index = 67)(d3)
    x56 = encoder.get_layer(index = 68)([x55, s1])
    x57 = encoder.get_layer(index = 69)(x56)
    x58 = encoder.get_layer(index = 70)(x57)
    x59 = encoder.get_layer(index =71)(x58)
    x60 = encoder.get_layer(index = 72)(x59)
    x61 = encoder.get_layer(index = 73)(x60)
    x62 = encoder.get_layer(index = 74)(x61)
    d4 = encoder.get_layer(index = 75)(x62)
    #_____________________
    outputs = encoder.get_layer(index = 76)(d4) 
    model = Model(inputs, outputs, name="U-Net")
    #setting all frozen layers from pretraining to trainable True
    for i in range(0,25):
        model.layers[i].trainable = True

    #setting batch normalization of encoder 1 trainable parameter to False
    model.layers[2].trainable = False
    model.layers[6].trainable = False
    #setting batch normalization of encoder 2 trainable parameter to False
    model.layers[10].trainable = False
    model.layers[14].trainable = False
    #setting batch normalization of encoder 3 trainable parameter to False
    model.layers[18].trainable = False
    model.layers[22].trainable = False

    i = 0
    for layer in model.layers:
        print(str(i) +": " + str(layer.name) + ", trainable: " + str(layer.trainable))
        print(layer.get_output_at(0).get_shape().as_list())
        i += 1
    return model

