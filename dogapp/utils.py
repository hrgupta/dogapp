# utils.py - utility functions to aid app operations.

import json
import os
import sys
import urllib.request

sys.path.append(".")
from datetime import datetime
from functools import wraps
from http import HTTPStatus

import numpy as np
from tensorflow.keras.preprocessing import image

from dogapp import models


# extract_VGG16, extract_VGG19, extract_Resnet50, extract_Xception, extract_InceptionV3 taken from erstwhile extract_bottleneck_features.py
def extract_VGG16(tensor):
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

    return VGG16(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_VGG19(tensor):
    from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

    return VGG19(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_Resnet50(tensor):
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    return ResNet50(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_Xception(tensor):
    from tensorflow.keras.applications.xception import Xception, preprocess_input

    return Xception(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


def extract_InceptionV3(tensor):
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

    return InceptionV3(weights="dogapp/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False).predict(
        preprocess_input(tensor)
    )


def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        with open("idata.jpg", "wb") as f:
            f.write(url.read())

    img_path = "idata.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    os.remove(img_path)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def get_run_components(url):

    # Load model
    model = models.DogCNN()
    model.summary(input_shape=(7, 7, 2048))  # build it
    model_path = os.path.join(os.getcwd(), "dogapp/weights.best.Xception.hdf5")
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
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url,
        }

        # Add data
        if results["status-code"] == HTTPStatus.OK:
            response["data"] = results["data"]

        return response

    return wrap
