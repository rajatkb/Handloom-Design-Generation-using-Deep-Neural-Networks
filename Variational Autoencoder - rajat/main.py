from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from keras.layers.advanced_activations import *
from keras import metrics

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from pylab import *
from drawnow import drawnow , figure

batch_size = 16
'''
 python main.py --lr 10e-4 --bs 16 --depoch 10 --gepoch 5 --epoch 1000

'''


class VarAutoEncoder:

    def make_trainable(self , net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def sampling(self , args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, self.noise_size),
                                  mean=self.epsilon_mean, stddev = self.epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def vae_loss(self , x, x_decoded_mean):
        xent_loss = K.mean(K.square(x - x_decoded_mean))
        kl_loss = - 0.5 * K.mean(1 + self.z_std - K.square(self.z_mean) - K.exp(self.z_std), axis=-1)
        return xent_loss + kl_loss


    def __init__(self , shape , lr=10e-4  ,noise_size=512):
        self.noise_size = noise_size ### the latent vector 
        assert(len(shape) == 3)
        self.channel = shape[2]
        
        self.losses={"g":[] , "d":[]}
        
        self.epsilon_std = 1
        self.epsilon_mean = 0
        opt = Adam(lr=lr)
        # dopt = SGD(lr=lr)
        
        inpe = Input(shape = shape , name = 'enc_input')
        layer = Conv2D(filters = 16 , kernel_size=(3,3) , strides=(3,3) , padding = 'valid' , kernel_initializer='glorot_uniform')(inpe)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters = 16 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)

        layer = Conv2D(filters = 32 , kernel_size=(3,3) , strides=(3,3) , padding = 'valid' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters = 32 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)

        layer = Conv2D(filters = 64 , kernel_size=(3,3) , strides=(2,2) , padding = 'valid' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters = 64 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)

        layer = Conv2D(filters = 128 , kernel_size=(3,3) , strides=(2,2) , padding = 'valid' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters = 128 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)

        layer = Conv2D(filters = 256 , kernel_size=(3,3) , strides=(1,1) , padding = 'valid' , kernel_initializer='glorot_uniform')(layer)
        layer = Flatten()(layer)
        layer = Dense(1024)(layer)

        self.z_mean = Dense(self.noise_size )(layer)
        self.z_std = Dense(self.noise_size )(layer) ## logarithmic standrad deviation

        self.encoder = Model(inputs=inpe , outputs=[self.z_mean,self.z_std])
        self.encoder.compile(loss = 'binary_crossentropy' , optimizer=opt) ## Makes no sense only to initialise model metrics
        self.encoder.summary()

        inpd = Input(shape=(self.noise_size,) , name='dec_input')
        layer = Dense(1024)(inpd)
        layer = Dense(4096)(layer)
        layer = Reshape((16 , 16 , 16 ))(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=128, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 128 , kernel_size=(5,5) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform' )(layer)
        layer = Conv2D(filters = 128 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 128 , kernel_size=(2,2) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=64, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 64 , kernel_size=(5,5) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform' )(layer)
        layer = Conv2D(filters = 64 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 64 , kernel_size=(2,2) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=32, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 32 , kernel_size=(5,5) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform' )(layer)
        layer = Conv2D(filters = 32 , kernel_size=(3,3) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 32 , kernel_size=(2,2) , strides=(1,1) , padding = 'same' , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=16, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters = 16 , kernel_size=(5,5) , strides=(1,1), padding='same',kernel_initializer='glorot_uniform' )(layer)
        layer = Conv2D(filters = 8 , kernel_size=(3,3) , strides=(1,1), padding='same',kernel_initializer='glorot_uniform' )(layer)
        layer = Conv2D(filters = 3 , kernel_size=(2,2) , strides=(1,1) , padding='same',kernel_initializer='glorot_uniform' )(layer)
        
        self.decoder = Model(inputs = inpd , outputs = layer)
        self.decoder.compile(loss = 'binary_crossentropy' , optimizer=opt)
        self.decoder.summary()

        z_mean , z_std = self.encoder(inpe)

        z = Lambda(self.sampling, output_shape=(self.noise_size,))([z_mean,z_std])

        vae_out = self.decoder(z)

        self.vae = Model(inputs=inpe , outputs=vae_out)
        # self.vae.load_weights('model.h5')
        self.vae.compile(loss = self.vae_loss , optimizer=opt)
        self.vae.summary()



    def draw_fig(self):
        display_index = np.random.randint(batch_size)
        subplot(2,2,1)
        noise = np.random.normal(self.data_mean,self.data_std,(1,self.noise_size))
        generated_images = self.decoder.predict(noise)
        # generated_images = np.array(generated_images * 255 , dtype=np.uint8)
        imshow(generated_images[0])
        subplot(2,2,2)
        generated_images = self.vae.predict(self.sample)
        # generated_images = np.array(generated_images * 255 , dtype=np.uint8)
        imshow(generated_images[display_index])
        subplot(2,2,3)
        imshow(self.sample[display_index])
        subplot(2,2,4)
        plot(self.losses["g"] , label="Loss")
        # plot(self.losses["d"] , label="discriminator")
        legend()



    # def gen_sample(self,epoch):
    #     noise = np.random.normal(0,1,size=[1,self.noise_size])
    #     generated_images = self.gen_model.predict(noise)
    #     generated_images = (generated_images * self.data_sd)+self.data_mean
    #     cv2.imwrite('result'+str(epoch)+'.jpg',generated_images[0])

    def train(self, data , epoch  ):
        self.data_mean = np.mean(data)
        self.data_std = np.std(data)
        # data = data /255
        self.sample = data[400:(400+batch_size)] ## because inference was not happening for the VAE
        for i in range(epoch):
            loss = self.vae.fit(x = data , y = data , batch_size = batch_size , epochs = 1 , verbose = 1 , shuffle = True )
            self.losses["g"]+=loss.history['loss']
            drawnow(self.draw_fig)
            self.vae.save_weights('model.h5')
            

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='data builder')
    parser.add_argument('--epoch', action="store",dest="epoch", default=100 )
    parser.add_argument('--lr', action="store",dest="learning_rate", default=10e-4)
    parser.add_argument('--data',action="store" , dest = "data" , default="../X.npy")
    # parser.add_argument('--bs' , action="store" , dest = "batch_size" , default=32)
    # parser.add_argument('--datay',action="store" , dest = "datay" , default="../Y.npy")

    values = parser.parse_args()
    epoch = int(values.epoch)
    learning_rate = float(values.learning_rate)
    data = np.load(values.data)

    data = data[:-(data.shape[0] % batch_size)]
    print("Data shape : ",data.shape)
    model = VarAutoEncoder(lr=learning_rate, shape = (data.shape[1],data.shape[2],data.shape[3]))
    model.train(data, epoch )
    
