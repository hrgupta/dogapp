# predict.py - predict (infer) inputs (single/batch).
import os
import sys
import collections
import json
import urllib.request
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from io import BytesIO
from tensorflow.keras.preprocessing import image

from argparse import ArgumentParser
from argparse import Namespace

from dogapp import config
from dogapp import utils
from dogapp import data
from dogapp import models

sys.path.append(".")

def get_run_components(url):

    # Load model
    model = models.DogCNN()
    model.summary(input_shape=(7,7,2048))  # build it
    model_path = os.path.join(os.getcwd() ,'dogapp/weights.best.Xception.hdf5')
    model.load_weights(model_path)
    data = loadImage(url)

    return url, data, model

def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        with open('idata.jpg', 'wb') as f:
            f.write(url.read())

    img_path = 'idata.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    os.remove(img_path)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def extract_Xception(tensor):
	from tensorflow.keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def predict(url, data, model):
    """Predict the class for a text using
    a trained model from an experiment."""

    #url = [sample['url'] for sample in inputs]

    with open(os.path.join(os.getcwd(), 'dogapp\dog_names.txt')) as f:
        dog_names = f.read().splitlines()

    # Predict
    results = []
    bottleneck_feature = extract_Xception(data)
    predicted_vector = model.predict(bottleneck_feature)
    p_class = dog_names[np.argmax(predicted_vector)]
    results.append({
        'input_url': url,
        'class': p_class,})
    return results


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--url', type=str,
                        required=True, help="image url to predict")
    args = parser.parse_args()
    inputs = [{'url': args.url}]

    # Get run components for prediction
    url, data, model = get_run_components(inputs[0]['url'])

    # Predict
    results = predict(url= url,data=data, model=model)
    config.logger.info(json.dumps(results, indent=4, sort_keys=False))