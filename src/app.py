import os
import tensorflow as tf
from keras_preprocessing import image
from tensorflow import keras
from flask import Flask, render_template, request, redirect

# set root path
ROOT_PATH = os.path.abspath('..')
MODEL_PATH = f'{ROOT_PATH}/models/best_model.h5'


# calling model
def get_model():
    global model
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded!")


# processing image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = tf.expand_dims(image.img_to_array(img), 0)
    return img_array  # prediction


def prediction(img_path):
    new_image = load_image(img_path)

    predictions = model.predict(new_image)
    score = tf.nn.softmax(predictions[0])

    if score[0] < 0.5:
        return "No Mask 😄"
    return "Mask 😷"


get_model()
app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and request.files['file']:
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(f'{ROOT_PATH}/src/static/uploads/', filename)  # slashes should be handled properly
        file.save(file_path)
        result = prediction(file_path)
        return render_template('predict.html', result=result, image=f"static/uploads/{filename}")
    return redirect("/", code=302)


if __name__ == "__main__":
    app.run()
