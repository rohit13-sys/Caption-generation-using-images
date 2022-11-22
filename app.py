import sys
import os
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from prediciton import *

IMAGE_FOLDER = os.path.join('static', 'photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


MODEL_PATH ='Image_caption_model.h5'
model = load_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')




def prediction_fn(file_path):
    Final_model = load_model('Image_caption_model.h5')
    path = file_path
    photo = extract_features(path)
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    max_length = 34
    description = generate_desc(Final_model, tokenizer, photo, max_length)
    print(description)
    return description



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = full_filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        result = prediction_fn(file_path)
        print(file_path)
        return render_template('index.html',result = result, image_path=file_path)
    return " "






if __name__ == '__main__':
    app.run(debug=True)
