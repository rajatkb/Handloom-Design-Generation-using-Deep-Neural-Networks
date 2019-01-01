from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from keras.layers.advanced_activations import *
from keras import metrics
from keras.applications import *
from keras.preprocessing import image
import tensorflow as tf

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


def residual_block(x , filters = 128):
    inp = x
    x = Lambda(lambda x: tf.pad(x , [[0,0],[1,1],[1,1],[0,0]], 'REFLECT'))(x)
    x = Conv2D(filters= filters , kernel_size=(3,3) , strides=(1,1) , padding='valid',kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: tf.pad(x , [[0,0],[1,1],[1,1],[0,0]], 'REFLECT'))(x)
    x = Conv2D(filters= filters , kernel_size=(3,3) , strides=(1,1) , padding='valid',kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Add()([inp , x])
    return x

def Convolution_block(x ,filters ,  kernel_size , strides , activation = 'relu'):
    # Use same padding when you are using stride greater than 1. 
    # The Calculation is not messed up then and you get expected block size
    p = ( kernel_size[0]-1 ) // 2
    if strides[0] == 1:
        x = Lambda(lambda x: tf.pad(x , [[0,0],[p,p],[p,p],[0,0]], 'REFLECT'))(x)
        x = Conv2D(filters= filters , kernel_size=kernel_size , strides=strides , padding='valid',kernel_initializer='glorot_uniform')(x)
    else:
        x = Conv2D(filters= filters , kernel_size=kernel_size , strides=strides , padding='same',kernel_initializer='glorot_uniform')(x)            
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def Convolution_transpose_block( x , filters ,  kernel_size , strides ):
    # Use same padding when you are using stride greater than 1. 
    # The Calculation is not messed up then and you get expected block size
    # x = UpSampling2D(size=strides)(x)
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size , strides=(2,2) , padding='same' , kernel_initializer='glorot_uniform')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x



inp = Input((256 , 256 , 3) , name='network_input')
x = Convolution_block(inp , filters =32 , kernel_size=(9,9) , strides=(1,1) )
x = Convolution_block(x , filters =64 , kernel_size=(3,3) , strides=(2,2) )
x = Convolution_block(x , filters =128 , kernel_size=(3,3) , strides=(2,2) )
x = residual_block(x , filters=128)
x = residual_block(x , filters=128)
x = residual_block(x , filters=128)
x = residual_block(x , filters=128)
x = residual_block(x , filters=128)
x = Convolution_transpose_block(x , filters=64 , kernel_size=(3,3) , strides=(2,2))
x = Convolution_transpose_block(x , filters=32 , kernel_size=(3,3) , strides=(2,2))
x = Convolution_block(x , filters=3 , kernel_size=(9,9) , strides=(1,1) , activation='tanh')
out = Lambda(lambda x: x*127.5 + 127.5 , name='scaling_output')(x)
gen_model = Model(inputs = inp , outputs = out)






def process_image(location):
	imagec = cv2.imread(location)
	image = cv2.imread(location , cv2.IMREAD_GRAYSCALE)
	image = cv2.resize(image , (256 , 256))
	imagec = cv2.resize(imagec , (256 , 256))
	assert(image is not None)
	# cv2.imshow('im1',image)
	(thresh, img) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	ind_w = img > 128
	ind_b = img < 128
	img[ind_w] = 0
	img[ind_b] = 255
	# cv2.imshow('im2',img)
	imgr = np.invert(img)
	# cv2.imshow('im3',imgr)
	# img = np.stack((img,)*3,-1)
	# imgr = np.stack((imgr,)*3,-1)
	return imagec , img , imgr

def transform(image , mask , model):
	gen_model.load_weights(model)
	image = gen_model.predict(np.array([image]))[0]
	image = cv2.bitwise_and(image,image , mask = mask)
	return image


import argparse
parser = argparse.ArgumentParser(description='merge style')
parser.add_argument('--inp',action="store",dest="input_image",required=True)
parser.add_argument('--style1' , action="store" , dest="model1",required=True)
parser.add_argument('--style2' , action="store" , dest="model2",required=True)
values = parser.parse_args()
input_image = values.input_image
model1 = values.model1
model2 = values.model2
image , mask , maskr = process_image(input_image)
im1 = transform(image , mask , model1)
im2 = transform(image , maskr , model2)
img = cv2.bitwise_or(im1 , im2 )
print(img)
plt.imshow(img/255)
plt.show()