
import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template ,Response
from flask import Flask, request,redirect,send_from_directory, render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

from PIL import Image


import cv2
import numpy as np
import pandas as pd
import argparse
#import tensorflow as tf
#from tensorflow import keras

import glob
import os

import random
from numpy import asarray
from itertools import repeat

import imageio
from imageio import imread
from PIL import Image
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model


import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify
from flask import send_file
import zipfile

#print("Tensorflow version " + tf.__version__)
#print("Keras version " + tf.keras.__version__)


outputimage = None
# output = []
# lock = threading.Lock()

def sample_images_test(img):

    
    lr_images = []
    
    img1 = imread(img, as_gray=False, pilmode='RGB')
    img1 = img1.astype(np.float32)
    
    img1_low_resolution = imresize(img1, (64, 64, 3))
          

    # do a random horizontal flip
    if np.random.random() < 0.5:
        img1_low_resolution = np.fliplr(img1_low_resolution)
    
    lr_images.append(img1_low_resolution)
        
   
    # convert lists into numpy ndarrays
    return  np.array(lr_images)    



def save_images_test(original_image , sr_image, path):
    
    """
    Save LR, HR (original) and generated SR
    images in one panel 
    """
    
    fig, ax = plt.subplots(1,2, figsize=(10, 6))

    images = [original_image, sr_image]
    titles = ['LR','SR']

    for idx,img in enumerate(images):
        # (X + 1)/2 to scale back from [-1,1] to [0,1]
        ax[idx].imshow((img + 1)/2.0, cmap='gray')
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
        
    plt.savefig(path)
    




# @st.cache
def fetch_model():
    return load_model('model.h5', compile=False)

loaded_model = fetch_model()

def pred_img(img):
    model = fetch_model()
    lr_img = sample_images_test(img)
    # normalize the images
    lr_img = (lr_img / 127.5) - 1
    generated_img = model.predict_on_batch(lr_img)
    lr_images_saved_m = lr_img.reshape((64,64,3))
    generated_images_saved_m= generated_img.reshape((256,256,3))
    save_images_test(lr_images_saved_m ,generated_images_saved_m ,"static/images/savedImage/outputimage.jpg")
    #path = "static/images/savedImage/outputimage.jpg"
	
    image =(generated_images_saved_m + 1)/2.0
    # st.image(image,clamp=True)
    return image

app = Flask(__name__)

@app.route('/upload')
def upload():
   return render_template('index.html')

@app.route('/')
def Rupload():
   return render_template('index.html')

@app.route('/index.html')
def home():
   return render_template('index.html')

@app.route('/inference.html')
def inference():
   return render_template('inference.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global outputimage
    if request.method == 'POST':
        f = request.files['Uplaod']
        #f.save(secure_filename(f.filename))
        f.save("static/images/savedImage/lr_image.jpg")
        predicted = pred_img("static/images/savedImage/lr_image.jpg")
        print(predicted)
        outputimage = cv2.imread("static/images/savedImage/outputimage.jpg",cv2.COLOR_BGR2RGB)
        print(type(predicted))

        print(outputimage)
        print(type(outputimage))
        return  render_template('inference.html')


def generate_feed():
    global outputimage
    (flag, encodedImage) = cv2.imencode(".jpg", outputimage)
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
        bytearray(encodedImage) + b'\r\n')



@app.route("/image_feed")
def image_feed():
    return Response(generate_feed(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")






def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='jpg') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img



@app.route('/get_images',methods=['GET'])
def get_images():

    ##reuslt  contains list of path images
    result = get_images_from_local_storage()
    encoded_imges = []
    for image_path in result:
        encoded_imges.append(get_response_image(image_path))
    return jsonify({'result': encoded_imges})




@app.route('/get_img')
def get_img():
    if request.args.get('type') == '1':
       filename = 'static/img/lr_image.jpg'
    else:
       filename = 'static/img/lr_image.jpg'
    return send_file(filename, mimetype='image/jpg')



# @app.route('/img_url')
# def img_url():
    
#     return '''<h1>The prediction is: {}</h1><h1>With a confidence of: {}%</h1>
#         <img src="{}" height = "85" width="200"/>'''.format(ok[0], okp, image_url)



if __name__ == '__main__':
   app.run(debug = True)
