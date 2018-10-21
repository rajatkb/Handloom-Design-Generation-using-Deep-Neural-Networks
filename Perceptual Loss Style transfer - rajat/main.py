from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from keras.layers.advanced_activations import *
from keras import metrics
from keras.applications import *
from keras.preprocessing import image

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

'''
 python main.py --lr 10e-4 --bs 16 --depoch 10 --gepoch 5 --epoch 1000

'''


class Fast_Style_transfer:

    def make_trainable(self , net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def residual_block(self , x , filters = 128):
        inp = x
        x = Conv2D(filters= filters , kernel_size=(3,3) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters= filters , kernel_size=(3,3) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Add()([inp , x])
        return x

    def Convolution_block(self , x ,filters ,  kernel_size , strides , padding , activation = 'relu'):
        # Use same padding when you are using stride greater than 1. 
        # The Calculation is not messed up then and you get expected block size
        x = Conv2D(filters= filters , kernel_size=kernel_size , strides=strides , padding=padding,kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def Convolution_transpose_block(self, x , filters ,  kernel_size , strides ):
        # Use same padding when you are using stride greater than 1. 
        # The Calculation is not messed up then and you get expected block size
        x = UpSampling2D(size=strides)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size , strides=(1,1) , padding='same' , kernel_initializer='glorot_uniform')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def get_vgg_loss_model(self,shape):
        vgg = vgg16.VGG16(input_shape=shape , weights='imagenet' , include_top=False)
        vgg_input = vgg.layers[0].output ## capturing input layer
        for l in vgg.layers: l.trainable=False
        ## Content loss
        self.content_loss_layer = 'block3_conv3'
        ## Texture loss
        self.texture_loss_layers = ['block1_conv2' , 'block2_conv2' , 'block3_conv3' , 'block4_conv3']

        content_loss_output = [vgg.get_layer(self.content_loss_layer).output]

        texture_loss_output = [vgg.get_layer(l).output for l in self.texture_loss_layers]

        model = Model(inputs=vgg_input , outputs = content_loss_output + texture_loss_output , name='vgg_loss')

        return model

    
    def set_constant_image(self, image , shape):
        img = cv2.imread(image)
        if img is None:
            raise Exception(' Bad image path given for '+image+' : Fix the name or the file does not exists : img value'+str(img))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img , (shape[0] , shape[1]))
        return Input(tensor = K.constant(np.array([img])))

    def get_content_loss(self,args):
        new_activation, content_activation = args[0], args[1]
        return K.constant(self.lambda_content) * K.mean(K.square(new_activation - content_activation))
    
    def gram_matrix(self , activation):
        shape = K.shape(activation)
        shape = (shape[0]*shape[1] , shape[2])
        # reshape to (H*W, C)
        activation = K.reshape(activation, shape)
        return K.dot( K.transpose(activation) , activation) / ( K.cast(shape[0],'float32') * K.cast(shape[1] , 'float32'))

    def get_texture_loss(self , args):
        new_activation, texture_activation = args[0], args[1]
        original_gram_matrix = self.gram_matrix(texture_activation[0])
        new_gram_matrix = self.gram_matrix(new_activation[0])
        return K.sum(K.square(original_gram_matrix - new_gram_matrix))

    def total_texture_loss(self , args):
        activations = args
        return K.constant(self.lambda_texture) * Add()(activations)

    def loss_model(self ,generator_input, generator_output , shape, content_image , texture_image):
        
        vgg = self.get_vgg_loss_model(shape)

        content_image = self.set_constant_image(content_image,shape)
        texture_image = self.set_constant_image(texture_image,shape)

        content_activations = vgg(content_image)
        texture_activations = vgg(texture_image)
        predicted_activations = vgg(generator_output) 

        content_loss = Lambda(self.get_content_loss,output_shape=(1,), name='content_loss')([predicted_activations[0], content_activations[0]])

        texture_losses = []
        for i in range(len(texture_activations)):
            texture_losses.append(Lambda(self.get_texture_loss , output_shape=(1,) , name='texture_loss'+str(i))([predicted_activations[i], texture_activations[i]]))

        texture_loss = Lambda(self.total_texture_loss , output_shape=(1,) , name='total_texture_loss')(texture_losses)

        model = Model(inputs=[generator_input,content_image,texture_image], outputs =[content_loss, texture_loss])

        return model

    def vgg_mean_sub(self , x):
        return K.bias_add(x ,K.constant(-np.array([123.68, 116.779, 103.939], dtype=np.float32)))

    def __init__(self ,content_image, texture_image, shape=(256,256,3) , lr=0.0001 ,chk = -1 , lambda_tv=0, lambda_content = 0 , lambda_texture = 0 , test=False):
        
        self.lambda_content = lambda_content
        self.lambda_texture = lambda_texture

        inp = Input(shape , name='network_input')
        x = self.Convolution_block(inp , filters =32 , kernel_size=(9,9) , strides=(1,1) , padding='same')
        x = self.Convolution_block(x , filters =64 , kernel_size=(3,3) , strides=(2,2) , padding='same')
        x = self.Convolution_block(x , filters =128 , kernel_size=(3,3) , strides=(2,2), padding='same')
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.residual_block(x , filters=128)
        x = self.Convolution_transpose_block(x , filters=64 , kernel_size=(3,3) , strides=(2,2))
        x = self.Convolution_transpose_block(x , filters=32 , kernel_size=(3,3) , strides=(2,2))
        out = self.Convolution_block(x , filters=3 , kernel_size=(9,9) , strides=(1,1) ,padding='same')
        gen_model = Model(inputs = inp , outputs = out)
        
        self.gen_model = gen_model

        gen_model.summary()

        if not test:
            
            ## VGG Preprocessing
            gen_model_output_vgg = Lambda(lambda x: K.reverse(x,axes=2) , name='flip_axis')(gen_model.output)
            gen_model_output_vgg = Lambda(lambda x: x * K.constant(255) , name='denormalise')(gen_model_output_vgg)
            gen_model_output_vgg = Lambda(self.vgg_mean_sub , name='vgg_normalise')(gen_model_output_vgg)
            ## VGG Preprocessing

            self.loss_model = self.loss_model(gen_model.input,gen_model_output_vgg,shape ,content_image , texture_image )
            self.loss_model.summary()
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None)
            self.loss_model.compile(loss={'content_loss':'mae' , 'total_texture_loss':'mae'}, optimizer = adam)

    def gen_sample(self,x,test_folder, epoch):
        generated_images = self.gen_model.predict(np.array([x])/255) # send in normalised
        img = cv2.cvtColor(generated_images[0]*255 , cv2.COLOR_BGR2RGB) # get denormalised result
        cv2.imwrite(test_folder+'/result'+str(epoch)+'.jpg',img)

    def generate_data(self , data_location):
        batches = os.listdir(data_location)
        for batch in batches:
            yield np.load(data_location+"/"+batch)

    def train(self, data_location ,test_folder, epochs  , batch_size):
        for epoch in range(epochs):
            for batch in self.generate_data(data_location):
                zero = np.zeros(batch.shape[0])

                ## send in normaised data
                norm_batch = batch/255
                self.loss_model.fit(x = norm_batch , y = {'content_loss':zero , 'total_texture_loss':zero} ,  batch_size = batch_size , verbose =1 , nb_epoch=1)
                
                img = cv2.cvtColor(batch[40] , cv2.COLOR_BGR2RGB)
                cv2.imwrite(test_folder+'/test_image.jpg',img)
                
                self.gen_sample(batch[40] , test_folder, epoch)
                self.gen_model.save_weights(test_folder+'/model_epoch'+str(epoch)+'.h5')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fast Neural Style Transfer')
    parser.add_argument('--test_name',action="store",dest="test_name",default="test_model")
    parser.add_argument('--bs', action="store",dest="batch_size" , default=8)
    parser.add_argument('--epoch', action="store",dest="epoch", default=100 )
    parser.add_argument('--lr', action="store",dest="learning_rate", default=10e-4)
    parser.add_argument('--data',action="store" , dest = "data" , default="../ms_coco_npy")
    parser.add_argument('--chk' , action="store" , dest = "chk" , default=-1)
    parser.add_argument('--lambda_content' , action="store" , dest = "lambda_content" , default=1e-6)
    parser.add_argument('--lambda_texture' , action="store" , dest="lambda_texture" , default= 1e-4)
    parser.add_argument('--content_image' , action="store" , dest = "content_image" , default="test_content.jpeg")
    parser.add_argument('--texture_image' , action="store" , dest = "texture_image" , default="test_texture.jpg")
    # parser.add_argument('--lamba_tv',action="store" , dest = "lambda_tv" , default=1e-6)


    values = parser.parse_args()
    epoch = int(values.epoch)
    batch_size = int(values.batch_size)
    learning_rate = float(values.learning_rate)
    chk = int(values.chk)
    lambda_content = float(values.lambda_content)
    lambda_texture = float(values.lambda_texture)
    content_image = values.content_image
    texture_image = values.texture_image
    test_folder = values.test_name
    data_location = values.data

    try:
        print("log: Creating test folder !!")
        os.mkdir(test_folder)
    except FileExistsError as e:
        print("log: Test folder exist")
    # data = data[:-(data.shape[0] % batch_size)]
    # print("Data shape : ",data.shape)
    model = Fast_Style_transfer(shape=(256,256,3) , lr=learning_rate ,chk = chk , lambda_tv=0, lambda_content = lambda_content , lambda_texture = lambda_texture ,content_image = content_image, texture_image = texture_image, test=False)
    model.train(data_location , test_folder,  epoch , batch_size)
    
