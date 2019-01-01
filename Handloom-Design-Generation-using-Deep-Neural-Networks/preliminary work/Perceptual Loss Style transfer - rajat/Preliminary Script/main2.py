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

'''
 python main.py --lr 10e-4 --bs 16 --depoch 10 --gepoch 5 --epoch 1000

 Difference from main.py

 The content image need not be set and more close to the original paper implementation
 Plus some minor changes of saving the model as Keras model and weights combined file

'''


class Fast_Style_transfer:

    def make_trainable(self , net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def residual_block(self , x , filters = 128):
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

    def Convolution_block(self , x ,filters ,  kernel_size , strides , activation = 'relu'):
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

    def Convolution_transpose_block(self, x , filters ,  kernel_size , strides ):
        # Use same padding when you are using stride greater than 1. 
        # The Calculation is not messed up then and you get expected block size
        # x = UpSampling2D(size=strides)(x)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size , strides=(2,2) , padding='same' , kernel_initializer='glorot_uniform')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def get_vgg_loss_model(self,shape):
        vgg = vgg16.VGG16(input_shape=shape , weights='imagenet' , include_top=False)
        vgg_input = vgg.layers[0].output ## capturing input layer
        for l in vgg.layers: l.trainable=False
        ## Content loss

        content_loss_output = [vgg.get_layer(self.content_loss_layer).output]

        texture_loss_output = [vgg.get_layer(l).output for l in self.texture_loss_layers]

        model = Model(inputs=vgg_input , outputs = content_loss_output + texture_loss_output , name='vgg_loss')

        return model

    def load_image(self , image_location , shape):
        if image_location is None:
            raise Exception('No image given for either content or style !!')
        img = cv2.imread(image_location)
        if img is None:
            raise Exception(' Bad image path given for '+image_location+' : Fix the name or the file does not exists : img value'+str(img))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , (shape[0] , shape[1]))
        return img
    
    def set_constant_image(self, image_location , shape):
        img = self.load_image(image_location , shape)
        return Input(tensor = K.constant(np.array([img])))

    def get_content_loss(self,args):
        new_activation, content_activation = args[0], args[1]
        return K.constant(self.lambda_content) * K.sum(K.square(new_activation - content_activation)) / 2
    
    def gram_matrix(self , activation):
        shape = K.shape(activation)
        shape = (shape[0]*shape[1] , shape[2])
        # reshape to (H*W, C)
        activation = K.reshape(activation, shape)
        return K.dot( K.transpose(activation) , activation) / (2 * K.cast(shape[0],'float32') * K.cast(shape[1] , 'float32'))

    def get_texture_loss(self , args):
        new_activation, texture_activation = args[0], args[1]
        original_gram_matrix = self.gram_matrix(texture_activation[0])
        new_gram_matrix = self.gram_matrix(new_activation[0])
        return (1/len(self.texture_loss_layers)) * K.sum(K.square(original_gram_matrix - new_gram_matrix))

    def total_texture_loss(self , args):
        activations = args
        t = K.constant(self.lambda_texture) * Add()(activations)
        return t

    def vgg_preprocess(self , image_tensor):
        ## VGG Preprocessing
        gen_model_output_vgg = Lambda(lambda x: K.reverse(x,axes=3))(image_tensor)
        # gen_model_output_vgg = Lambda(lambda x: x*127.5 + 127.5 , name='vgg_scaling')(gen_model_output_vgg)
        gen_model_output_vgg = Lambda(lambda x: K.bias_add(x ,K.constant(-np.array([ 103.939, 116.779,123.68], dtype=np.float32))))(gen_model_output_vgg)
        ## VGG Preprocessing
        return gen_model_output_vgg

    def total_variation_loss(self , x):
        img_nrows , img_ncols , _ = self.shape
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        r = self.lambda_tv * K.sum(K.pow(a + b, 1.25))
        return r


    def loss_model(self , gen_model , shape , texture_image_location):
        
        vgg = self.get_vgg_loss_model(shape)

        generator_input = gen_model.input
        generator_output = gen_model.output

        # content_image = self.set_constant_image(content_image_location,shape)
        content_image_preprocessed = self.vgg_preprocess(generator_input)

        texture_image = self.set_constant_image(texture_image_location,shape)
        texture_image_preprocessed = self.vgg_preprocess(texture_image)
        predicted_image_preprocessed = self.vgg_preprocess(generator_output)

        content_activations = vgg(content_image_preprocessed)
        texture_activations = vgg(texture_image_preprocessed)
        predicted_activations = vgg(predicted_image_preprocessed) 

        content_loss = Lambda(self.get_content_loss,output_shape=(1,), name='content_loss')([predicted_activations[0], content_activations[0]])

        texture_losses = []
        for i in range(1,len(texture_activations)):
            texture_losses.append(Lambda(self.get_texture_loss , output_shape=(1,) , name='texture_loss'+str(i))([predicted_activations[i], texture_activations[i]]))

        texture_loss = Lambda(self.total_texture_loss , output_shape=(1,) , name='total_texture_loss')(texture_losses)

        total_variation_loss = Lambda(self.total_variation_loss , output_shape=(1,) , name='total_variation_loss')(generator_output)

        model = Model(inputs=[generator_input,texture_image], outputs =[ total_variation_loss, content_loss, texture_loss , generator_output])

        return model


    def __init__(self , texture_image=None, test_folder=None , shape=(256,256,3) , lr=0.0001 ,chk = -1 , lambda_tv=0, lambda_content = 0 , lambda_texture = 0 , test=False, model_name="model.h5"):
        
        self.shape = shape
        self.checkpoint = chk
        self.lambda_content = lambda_content
        self.lambda_texture = lambda_texture
        self.lambda_tv = lambda_tv
        self.test_folder = test_folder

        self.content_loss_layer = 'block2_conv2'
        ## Texture loss
        self.texture_loss_layers = [
                                    'block1_conv1' , 
                                    'block2_conv1' , 
                                    'block3_conv1' , 
                                    'block4_conv1',
                                    'block5_conv1'
                                    ]

        inp = Input(shape , name='network_input')
        x = Lambda(lambda x: x/127.5 - 1 , name = 'scaling_input')(inp)
        x = self.Convolution_block(x , filters =32 , kernel_size=(9,9) , strides=(1,1) )
        x = self.Convolution_block(x , filters =64 , kernel_size=(3,3) , strides=(2,2) )
        x = self.Convolution_block(x , filters =128 , kernel_size=(3,3) , strides=(2,2) )
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.Convolution_transpose_block(x , filters=64 , kernel_size=(3,3) , strides=(2,2))
        x = self.Convolution_transpose_block(x , filters=32 , kernel_size=(3,3) , strides=(2,2))
        x = self.Convolution_block(x , filters = 32 , kernel_size = (3 , 3) , strides = (1,1))
        x = self.Convolution_block(x , filters=3 , kernel_size=(9,9) , strides=(1,1) , activation='tanh')
        out = Lambda(lambda x: x*127.5 + 127.5 , name='scaling_output')(x)
        gen_model = Model(inputs = inp , outputs = out)
        
        self.gen_model = gen_model

        if test:
            gen_model.load_weights(model_name)
        
        if not test:
            gen_model.summary()    
            self.loss_model = self.loss_model(gen_model,shape  , texture_image )
            self.loss_model.summary()
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None)
            self.loss_model.compile(loss={'content_loss':'mae' , 'total_texture_loss':'mae' ,'total_variation_loss':'mae' , 'scaling_output':'mse'}, optimizer = adam)
            self.gen_model.compile(loss='mae' , optimizer = adam)
            if chk > -1:
                self.gen_model.load_weights(test_folder+'/model_epoch'+str(chk)+'.h5')

    def gen_sample(self,x,test_folder, epoch):
        generated_images = self.gen_model.predict(np.array([x])) # send in normalised
        img = cv2.cvtColor(generated_images[0] , cv2.COLOR_BGR2RGB) # get denormalised result
        cv2.imwrite(test_folder+'/result'+str(epoch)+'.jpg',img)

    def generate_image(self , x):
        generated_images = self.gen_model.predict(np.array([x]))
        img = cv2.cvtColor(generated_images[0] , cv2.COLOR_BGR2RGB) # get denormalised result
        cv2.imwrite('result.jpg',img)

    def generate_data(self , data_location):
        batches = os.listdir(data_location)
        for batch in batches:
            yield np.load(data_location+"/"+batch)

    def train(self, data_location , epochs  , batch_size):
        if self.checkpoint < 0:
            self.checkpoint = 0
        else:
            self.checkpoint +=1
        test_folder = self.test_folder
        for epoch in range(self.checkpoint,epochs):
            print("log: epoch running:",epoch)
            for batch in self.generate_data(data_location):
                
                if epoch <= 1:
                    print("log: fitting network to generate images !! epoch:",epoch)
                    self.gen_model.fit(x = batch , y = batch , batch_size=4 , verbose = 1 , epochs=20)

                if epoch > 1:
                    zero = np.zeros(batch.shape[0])
                    print("log: fitting network for style and content")
                    self.loss_model.fit(x = batch , y = {'content_loss':zero , 'total_texture_loss':zero ,'total_variation_loss':zero , 'scaling_output':batch} ,  batch_size = batch_size , verbose =1 , epochs=1)
                
                img = cv2.cvtColor(batch[200] , cv2.COLOR_BGR2RGB) ## Becuase while saving this operation is run again by CV library
                cv2.imwrite(test_folder+'/test_image.jpg',img)
                
                self.gen_sample(batch[200] , test_folder, epoch)
                self.gen_model.save_weights(test_folder+'/model_epoch'+str(epoch)+'.h5')
                self.gen_model.save(test_folder+"/gen_model.h5")


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fast Neural Style Transfer')
    parser.add_argument('--exp_name',action="store",dest="train_name",default="test_model")
    parser.add_argument('--bs', action="store",dest="batch_size" , default=8)
    parser.add_argument('--epoch', action="store",dest="epoch", default=100)
    parser.add_argument('--lr', action="store",dest="learning_rate", default=10e-4)
    parser.add_argument('--data',action="store" , dest = "data" , default="../ms_coco_npy")
    parser.add_argument('--chk' , action="store" , dest = "chk" , default=-1)
    parser.add_argument('--lambda_content' , action="store" , dest = "lambda_content" , default=1e-6)
    parser.add_argument('--lambda_texture' , action="store" , dest="lambda_texture" , default= 1e-4)
    parser.add_argument('--texture_image' , action="store" , dest = "texture_image" , default="test_texture.jpg")
    parser.add_argument('--lambda_tv',action="store" , dest = "lambda_tv" , default=1e-6)
    parser.add_argument('--test' , action='store' , dest = 'test' , default=False)
    parser.add_argument('--model' , action='store' , dest = 'model' , default='model.h5')
    parser.add_argument('--test_image' , action ='store' , dest ='test_image' , default='image.png')
    values = parser.parse_args()
    epoch = int(values.epoch)
    batch_size = int(values.batch_size)
    learning_rate = float(values.learning_rate)
    chk = int(values.chk)
    lambda_content = float(values.lambda_content)
    lambda_texture = float(values.lambda_texture)
    texture_image = values.texture_image
    train_folder = values.train_name
    data_location = values.data
    lambda_tv = float(values.lambda_tv)
    test = values.test
    model_name = values.model
    test_image = values.test_image

    if not test:
        try:
            print("log: Creating test folder !!")
            os.mkdir(train_folder)
        except FileExistsError as e:
            print("log: Test folder exist")
    
    model = Fast_Style_transfer(shape=(256,256,3) , lr=learning_rate ,chk = chk , lambda_tv=lambda_tv, lambda_content = lambda_content , lambda_texture = lambda_texture , texture_image = texture_image, test_folder=train_folder ,test=test , model_name=model_name)
    
    if not test:
        model.train(data_location ,  epoch , batch_size)

    if  test:
        img = model.load_image(test_image, (256,256,3))
        model.generate_image(img)

    
