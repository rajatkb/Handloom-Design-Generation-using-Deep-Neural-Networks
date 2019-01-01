'''
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)
global graph
graph = tf.get_default_graph()
'''
from keras import backend as K



from flask_cors import CORS
from flask import Flask, render_template , request , jsonify, Response
from PIL import Image
import scipy
from glob import glob
import numpy as np 
import cv2
import base64
from discogan_final import Discogan
from data_loader import DataLoader
from keras.layers import Dense, Activation
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import io
import os
import random


app=Flask(__name__)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@app.route('/test',methods=['GET','POST'])
def test():
    print("log:secces")
    return jsonify({'status': 'success'})
@app.route('/genColor')
def home():
    
    return render_template('index.jinja2')

@app.route('/maskImage' , methods=['POST'])
def discogan_mask_generate_image():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img1 = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    
    lab= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    
    im_gray = cl
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img=np.stack((im_bw,)*3,-1)
    
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

@app.route('/colorImage' , methods=['POST'])
def discogan_color_generate_image():
    K.clear_session()
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img1 = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    
    lab= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    r, c = 1,1
    img_rows = 256
    img_cols = 256
    channels = 3
    #img_shape = (img_rows, img_cols, channels)
    
    
    im_gray = cl
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img=np.stack((im_bw,)*3,-1)
    #edge_bw = cv2.resize(img , (256 , 256))
    cv2.imwrite("img.jpg",img)
    data_loader = DataLoader(dataset_name="img.jpg",
                              img_res=(256, 256))


    
    imgs_A = data_loader.load_data(batch_size=1, is_testing=True)
    
    model = load_model('saved_model/actual_model16.h5')
    optimizer = Adam(0.0002, 0.5)

    model.compile(loss='mse',optimizer=optimizer)

    # Translate images to the other domain
    print(imgs_A.shape)
    img = model.predict(imgs_A)
    #cv2.imwrite("img2.jpg",img[0])
    gen_imgs = np.concatenate([img])
    fig, axs = plt.subplots(r, c)
    axs.imshow(gen_imgs[0])
    
    axs.axis('off')

    fig.savefig("img.jpg")
    plt.close()
    with open("img.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return jsonify({'status':str(encoded_string)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9000,debug=True)