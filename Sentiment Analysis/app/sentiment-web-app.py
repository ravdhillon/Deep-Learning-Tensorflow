import tensorflow as tf
import tensorflow_datasets as tfds

from flask import Flask, jsonify, make_response, request

print('Starting web app')
# create Flask application and assign to app object.
app = Flask(__name__)
padding_size = 1000
model = tf.keras.models.load_model('../model/sentiment_analysis.hdf5')
text_encoder = tfds.features.text.TokenTextEncoder.load_from_file('../model/sa_encoder.vocab.tokens')

print('Model and Vocabulary loaded...')

def pad_to_size(vec, size):
    zeros = [0] * (len(vec) - size)
    vec.extend(zeros)
    return vec

def predict(text, pad_size):
    encoded_text = text_encoder.encode(text)
    encoded_text = pad_to_size(encoded_text, pad_size)
    encoded_text = tf.cast(encoded_text, tf.int64)
    predictions = model.predict(tf.expand_dims(encoded_text, 0))

    return (predictions.tolist())


@app.route('/classifier', methods=['POST'])
def predict_sentiment():
    text = request.get_json()['text']
