# Hosted deploy entry point (Streamlit Cloud) — self-contained Streamlit UI
# with direct inference (no FastAPI backend; the host runs a single process).
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import streamlit as st

from dogapp import dog, models, predict, utils

EMBEDDINGS_PATH = "embeddings/weights.best.Xception.hdf5"


@st.cache_resource
def load_model():
    model = models.DogCNN()
    model.summary(input_shape=(7, 7, 2048))  # build it
    model.load_weights(EMBEDDINGS_PATH)
    return model


def get_model_output(url):
    """Direct prediction — mirrors the FastAPI /predict response shape."""
    model = load_model()
    data = utils.loadImage(url)
    prediction = predict.predict(url=url, data=data, model=model)[0]
    return {"data": {"prediction": [prediction]}}


# Point the existing Streamlit UI at the direct predictor.
dog.get_model_output = get_model_output
dog.main()
