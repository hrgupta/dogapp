import os
import sys
sys.path.append(".")
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

from dogapp import config
from dogapp import predict
from dogapp import utils

# Title
st.title("Dog Identification 🐶 App")

# Get run components for prediction
#url, data, model = predict.get_run_components(payload.inputs[0]['url'])

# Pages
#page = st.sidebar.selectbox(
#    "Choose a page", ['Dog Identification 🐶 App', 'Other Apps..'])

st.header("🚀 Try it out!")

text = st.text_input(
    "Enter dog image url:", value="https://news.nationalgeographic.com/content/dam/news/2018/05/29/dog-baby-talk/01-dog-baby-talk-NationalGeographic_2283205.ngsversion.1527786004161.adapt.1900.1.jpg")

st.image(text, use_column_width=True)

url, data, model = predict.get_run_components(text)

results = predict.predict(url= url,data=data, model=model)

raw_text = results[0]['input_url']
st.write("**Input url**:", raw_text)
preprocessed_text = results[0]['class']
st.write("**Predicted breed**:", preprocessed_text)

# Warning
#st.info("""⚠️The model architecture used in this demo is **not state-of-the-art** and it's not **fully optimized**, as this was not the main focus of the lesson.
#        Also keep in mind that the **dataset** is dated and **limited** to particular vocabulary.
#        If you are interested in generalized text classification or NLP in general, check out these [curated resources](https://madewithml.com/topics/#nlp).""")

# Show raw json
show_json = st.checkbox("Show complete JSON output:", value=False)
if show_json:
    st.json(results)
    
#if page == 'Prediction':
#    pass

    # Input text
    
    # Get Run components


    # Predict

    
    # Results
#elif page == 'Model details':

#    st.header("Other Applications")
    
#    st.write("More apps coming soon...")