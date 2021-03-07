# predict.py - predict (infer) inputs (single/batch).
import os
import sys

sys.path.append(".")  # necessary to run predict.py from command line *magic line*
import collections
import json
from argparse import ArgumentParser, Namespace
from io import BytesIO

import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from dogapp import config, data, models, utils

sys.path.append(".")


def predict(url, data, model):
    """Predict the class for a text using
    a trained model from an experiment."""

    # url = [sample['url'] for sample in inputs]

    with open(os.path.join(os.getcwd(), "dogapp/dog_names.txt")) as f:
        dog_names = f.read().splitlines()

    # Predict
    results = []
    bottleneck_feature = utils.extract_Xception(data)
    predicted_vector = model.predict(bottleneck_feature)
    p_class = dog_names[np.argmax(predicted_vector)]
    results.append(
        {
            "input_url": url,
            "class": p_class,
        }
    )
    return results


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="image url to predict")
    args = parser.parse_args()
    inputs = [{"url": args.url}]

    # Get run components for prediction
    model = models.DogCNN()
    model.summary(input_shape=(7, 7, 2048))  # build it
    model_path = os.path.join(os.getcwd(), "embeddings/weights.best.Xception.hdf5")
    model.load_weights(model_path)
    data = utils.loadImage(inputs[0]["url"])

    # Predict
    prediction = []
    prediction.append(predict(url=inputs[0]["url"], data=data, model=model)[0])

    config.logger.info(json.dumps(prediction, indent=4, sort_keys=False))
