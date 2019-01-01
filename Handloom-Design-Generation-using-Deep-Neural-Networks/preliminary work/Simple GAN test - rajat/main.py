from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as k
from keras.layers.advanced_activations import *

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from pylab import *
from drawnow import drawnow , figure

class DCGAN:

    def make_trainable(self , net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val


    def __init__(self , shape , lr=10e-4 , epoch=100 ,noise_size=64):
        self.noise_size = noise_size
        assert(len(shape) == 3)
        self.channel = shape[2]
        
        self.losses={"g":[] , "d":[]}
        
        opt = Adam(lr=lr)
        dopt = SGD(lr=lr)
        
        inpg = Input(shape = (self.noise_size,) , name='gen_input')
        layer = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(inpg)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(256  , activation='relu', kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(768  , activation='relu', kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Reshape((16, 16, 3))(layer)
        layer = Conv2DTranspose(filters=256, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,activation='relu',kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters=256 , kernel_size=(3,3),strides=(1,1) , padding='same' ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=256 , kernel_size=(1,1),strides=(1,1) , padding='same' ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=128, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,activation='relu', kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters=128 , kernel_size=(3,3),strides=(1,1) , padding='same' ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=128 , kernel_size=(1,1),strides=(1,1) , padding='same' ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=64, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,activation='relu',kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters=64 , kernel_size=(3,3),strides=(1,1) , padding='same'  ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=64 , kernel_size=(1,1),strides=(1,1) , padding='same'  ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=32, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,activation='relu', kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters=32 , kernel_size=(3,3),strides=(1,1) , padding='same'  ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=32 , kernel_size=(1,1),strides=(1,1) , padding='same'  ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2DTranspose(filters=3, kernel_size=(2,2) , strides=(2,2) , padding='valid' ,activation='relu',kernel_initializer='glorot_uniform')(layer)
        layer = Conv2D(filters=3 , kernel_size=(3,3),strides=(1,1) , padding='same'   ,kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=3 , kernel_size=(1,1),strides=(1,1) , padding='same' ,activation='tanh'  ,kernel_initializer='glorot_uniform')(layer)
        self.gen_model = Model(inputs = inpg , outputs = layer)
        self.gen_model.compile(loss = 'binary_crossentropy' , optimizer=opt)
        self.gen_model.summary()
        
        
        inpd = Input(shape = shape , name='disc_input')
        layer = Conv2D(filters= 16 , kernel_size=(7,7) , strides=(1,1) , padding='valid'  , kernel_initializer='glorot_uniform')(inpd)
        layer = LeakyReLU(0.3)(layer)
        layer = Conv2D(filters= 32 , kernel_size=(5,5) , strides=(1,1) , padding='valid'  , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.3)(layer)
        layer = AveragePooling2D(pool_size=(5,5) , padding='valid')(layer)
        layer = Conv2D(filters= 64 , kernel_size=(1,1) , strides=(2,2) , padding='valid'  , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.3)(layer)
        layer = Conv2D(filters= 128 , kernel_size=(1,1) , strides=(2,2) , padding='valid'  , kernel_initializer='glorot_uniform')(layer)
        layer = LeakyReLU(0.3)(layer)
        layer = AveragePooling2D(pool_size=(3,3) , padding='valid')(layer)
        layer = Flatten()(layer)
        layer = Dense(64 , kernel_initializer='glorot_uniform' )(layer)
        layer = LeakyReLU(0.5)(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(32 , kernel_initializer='glorot_uniform' )(layer)
        layer = LeakyReLU(0.3)(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(16 , kernel_initializer='glorot_uniform' )(layer)
        layer = Dense(1 , activation='sigmoid', kernel_initializer='glorot_uniform')(layer)
        self.disc_model = Model(inputs = inpd , outputs = layer)
        self.disc_model.compile(loss = 'binary_crossentropy' , optimizer=dopt , metrics=['acc'])
        self.disc_model.summary()
        
        self.make_trainable(self.disc_model , False)

        gan_inp = Input(shape = (self.noise_size,) , name='gan_input')
        gan_out = self.disc_model(self.gen_model(gan_inp))
        self.DCGAN = Model(inputs = gan_inp , outputs = gan_out)
        # if os.path.isfile('model.h5'):
        self.DCGAN.load_weights('model.h5')
        self.DCGAN.compile(loss = 'binary_crossentropy' , optimizer = opt , metrics=['acc'])
        self.DCGAN.summary()


    def draw_fig(self):
        subplot(1,2,1)
        noise = np.random.normal(0,1,size=[1,self.noise_size])
        generated_images = self.gen_model.predict(noise)
        generated_images = (generated_images * self.data_sd)+self.data_mean
        imshow(generated_images[0])

        subplot(1,2,2)
        plot(self.losses["g"] , label="generated")
        plot(self.losses["d"] , label="discriminator")
        legend()



    # def gen_sample(self,epoch):
    #     noise = np.random.normal(0,1,size=[1,self.noise_size])
    #     generated_images = self.gen_model.predict(noise)
    #     generated_images = (generated_images * self.data_sd)+self.data_mean
    #     cv2.imwrite('result'+str(epoch)+'.jpg',generated_images[0])

    def train(self, data , epoch , learning_rate , batch_size=32 , disc_epoch=3 , gan_epoch=2):

        self.data_mean = np.mean(data)
        self.data_sd = np.std(data) 
        data = (data - self.data_mean)/self.data_sd ## scaling the data to match sigmoid
        gan_label = np.ones(data.shape[0]*2)

        for i in range(epoch):
            print("Epoch :" , i)
            noise_gen = np.random.normal(0, 1, (data.shape[0],self.noise_size))
            gen_data = self.gen_model.predict(noise_gen,verbose=1) 
            disc_train_data = np.concatenate([data,gen_data],axis=0)
            disc_label = np.zeros(data.shape[0]*2)
            disc_label[:data.shape[0]] = 1
            self.make_trainable(self.disc_model , True)
            d_loss = self.disc_model.fit(x=disc_train_data, y=disc_label, batch_size=batch_size, epochs=disc_epoch, verbose=1 , shuffle=True)
            del(disc_train_data)
            del(noise_gen)
            del(gen_data)
            self.losses["d"]+=d_loss.history['loss']
            noise_gen = np.random.normal(0, 1, (data.shape[0]*2,self.noise_size))
            self.make_trainable(self.disc_model , False)
            g_loss = self.DCGAN.fit(x=noise_gen,y=gan_label , batch_size=batch_size ,epochs=gan_epoch, verbose=1 , shuffle=True )
            del(noise_gen)
            self.losses["g"]+=g_loss.history['loss']
            drawnow(self.draw_fig)
            if i%5 == 0:
                # self.gen_sample(i)
                self.DCGAN.save_weights('model.h5')
            

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='data builder')
    parser.add_argument('--epoch', action="store",dest="epoch", default=100 )
    parser.add_argument('--depoch', action="store",dest="depoch", default=3 )
    parser.add_argument('--gepoch', action="store",dest="gepoch", default=2 )
    parser.add_argument('--lr', action="store",dest="learning_rate", default=10e-4)
    parser.add_argument('--data',action="store" , dest = "data" , default="../Y.npy")
    parser.add_argument('--bs' , action="store" , dest = "batch_size" , default=32)
    # parser.add_argument('--datay',action="store" , dest = "datay" , default="../Y.npy")

    values = parser.parse_args()
    epoch = int(values.epoch)
    learning_rate = float(values.learning_rate)
    data = np.load(values.data)
    batch_size = int(values.batch_size)
    depoch = int(values.depoch)
    gepoch = int(values.gepoch)

    model = DCGAN(shape = (512,512,3))
    model.train(data, epoch, learning_rate , batch_size , disc_epoch=depoch , gan_epoch=gepoch)
    
