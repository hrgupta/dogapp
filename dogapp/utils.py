# utils.py - utility functions to aid app operations.

import json
import os
import sys
import time
import urllib.request

sys.path.append(".")
from datetime import datetime
from functools import wraps
from http import HTTPStatus

import numpy as np
from keras.utils import load_img, img_to_array

from dogapp import models


# extract_VGG16, extract_VGG19, extract_Resnet50, extract_Xception, extract_InceptionV3 taken from erstwhile extract_bottleneck_features.py
def extract_VGG16(tensor):
    from keras.applications.vgg16 import VGG16, preprocess_input

    return VGG16(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_VGG19(tensor):
    from keras.applications.vgg19 import VGG19, preprocess_input

    return VGG19(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input

    return ResNet50(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input

    return Xception(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input

    return InceptionV3(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def loadImage(URL, retries=3):
    # Wikimedia and other hosts reject bare urllib requests — send a real UA.
    # Some hosts also throttle/reset mid-download, so retry with backoff.
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL,
                headers={"User-Agent": "dogapp/1.0 (https://github.com/hrgupta/dogapp)"},
            )
            with urllib.request.urlopen(req, timeout=30) as url:
                with open("idata.jpg", "wb") as f:
                    f.write(url.read())
            break
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    else:
        raise last_err

    img_path = "idata.jpg"
    img = load_img(img_path, target_size=(224, 224))
    os.remove(img_path)
    x = img_to_array(img)
    return np.expand_dims(x, axis=0)


def get_run_components(url):

    # Load model
    model = models.DogCNN()
    model.summary(input_shape=(7, 7, 2048))  # build it
    model_path = os.path.join(os.getcwd(), "embeddings/weights.best.Xception.hdf5")
    model.load_weights(model_path)
    data = loadImage(url)

    return url, data, model


def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(*args, **kwargs):
        results = f(*args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
        }

        # Add data
        if results["status-code"] == HTTPStatus.OK:
            response["data"] = results["data"]

        return response

    return wrap
