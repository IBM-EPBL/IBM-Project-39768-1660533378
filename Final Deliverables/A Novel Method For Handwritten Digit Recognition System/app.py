from flask import Flask, request, render_template
from PIL import Image
import numpy as np
#from requests import post
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from io import BytesIO

app = Flask('__main__')

#model = load_model('models/mnistCNN.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('web.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method =='POST':
        img = request.files['file']
        img = Image.open(img.stream).convert('L')
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = model.predict(im2arr)
        y_pred = np.argmax(y_pred,axis=1)
        print(y_pred)
        return render_template('web.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
    0