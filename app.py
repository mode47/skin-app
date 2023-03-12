from tensorflow.keras.applications import VGG19, EfficientNetB0, VGG16, InceptionV3, ResNet50, EfficientNetB3
import tensorflow as tf
from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
app = Flask(__name__)


def predict_image_class(image):
    image = Image.open(image)
    image = image.resize((180, 180))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 180, 180, 3)
    features_train = vgg_model.predict(image_arr)
    # print(features_train.shape)
    img = features_train.reshape(1, -1)
    y_pred = model.predict(img)
    y_pred2 = np.argmax(y_pred, axis=1)
    return y_pred2


vgg_model = VGG19(
    weights='imagenet',  include_top=False, input_shape=(180, 180, 3))
#classes = ['Buildings' ,'Forest', 'Glacier' ,'Mountain' ,'Sea' ,'Street']
classes = ['Acne',
           'Normal',
           'vitiligo',
           'Tinea',
           'Melanoma Skin Cancer',
           'Eczema Photos']
# model=load_model(r"C:\Users\hp\Downloads\intel-Classfier-main\intel-Classfier-main\Intel_Image_Classification.h5")




model = load_model('6claass (3).h5')


@app.route('/')
def index():
    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("Model predicted")
        #ind = np.argmax(result)
        ind = predict_image_class(image)
        prediction = classes[int(ind)]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print(image)
        ind = predict_image_class(image)
        prediction = classes[int(ind)]
        print(prediction)
        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html', appName="Intel Image Classification")


if __name__ == '__main__':
    app.run(debug=True)
