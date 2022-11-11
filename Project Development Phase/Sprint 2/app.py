from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from requests import post
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt 
from io import BytesIO
import base64

app = Flask('__main__')

model = load_model('models/mnistCNN.h5')


def generate(y_pred):
    n_o_c = 10
    y_pred = np_utils.to_categorical(y_pred,n_o_c)
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = y_pred.astype(int)
    y = y[0]
    ch = np.where(y ==1)
    y[ch] = 10
    data = sub(x,y)
    return data

def sub(x,y):
    plt.bar(x,y, color = 'red')
    b = BytesIO()
    plt.savefig(b, format='png')
    data = base64.b64encode(b.getbuffer()).decode()
    return data

def show(img):
    img = img
    data = base64.b64encode(img.getbuffer()).decode()
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    plt.clf()
    return render_template('web.html',show = 'hidden')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method =='POST':
        img = request.files['file']
        data = show(img)
        img = Image.open(img.stream).convert('L')
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = model.predict(im2arr)
        y_pred = np.argmax(y_pred,axis=1)
        pred = str(y_pred)
        output = "Recognized digit is :"+ pred
        bar = generate(y_pred)
        return render_template('web.html',output = output,bar = bar,data = data,button = 'hidden' )



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
    