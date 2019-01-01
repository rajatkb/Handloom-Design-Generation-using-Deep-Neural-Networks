from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)
global graph
graph = tf.get_default_graph()

from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from flask_cors import CORS
from utils import create_mask , get_masks , merge

import nets

dim = 512
folder = "saved_weights/"
model = nets.image_transform_net(dim , dim , 1e-6)
model.load_weights(folder+"colorfull.h5")
app = Flask(__name__)

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# # cors = CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app)

@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})

# @app.route('/home')
# def home():
# 	return render_template('index.jinja2')



@app.route('/maskImage' , methods=['POST'])
def mask_image():
	print("log: recieved masking rqeuest", file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
	
    ######### Do preprocessing here ################
	img = create_mask(img , channel = 3)
	################################################
	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})
	    
@app.route('/mergeStyle' , methods = ['POST'])
def merge_style():
    print("log: recieved merge request" , file=sys.stderr)
    background = request.form['background']
    foreground = request.form['foreground']
    file = request.files['image'].read() ## byte file
    npimg = np.fromstring(file , np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img , (dim , dim))
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    ############# Processing here ###########################
    with graph.as_default():
        img = merge(img , model , dir="saved_weights/" , weights = [foreground, background])
    #########################################################
    
    img = Image.fromarray(img.astype('uint8'))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})
    
    
if __name__ == '__main__':
	app.run(debug=True , port=7000 ,host='0.0.0.0')