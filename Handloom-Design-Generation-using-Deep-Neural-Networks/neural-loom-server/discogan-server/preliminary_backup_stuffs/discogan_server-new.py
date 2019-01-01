from flask import Flask, render_template , request , jsonify, Response
from PIL import Image
import scipy
from glob import glob
import numpy as np 
import cv2
import base64
import tensorflow as tf
import keras as K
from discogan_final import Discogan
from data_loader import DataLoader
from keras.layers import Dense, Activation , Input
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import io
import os
import random

global graph
graph = tf.get_default_graph()
dimension = 256
model = load_model('saved_model/actual_model16.h5')
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


@app.route('/colorImage' , methods=['POST'])
def discogan_color_generate_image():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img1 = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1 , (dimension , dimension))
    ######################## Processing image ##########################
    lab= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    im_gray = cl
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img=np.stack((im_bw,)*3,-1)
    img=np.array([img])
    with graph.as_default():
        img=model.predict(img)[0]
    
    #cv2.imwrite("img.jpg",img[0])
    ############################# processing done #############################
    
    img = Image.fromarray(img[0].astype("uint8"))
    rawBytes = io.BytesIO()
    
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=7000,debug=True)