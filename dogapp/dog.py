import os
import sys
from typing import List

sys.path.append(".")
import json

import numpy as np
import requests
import streamlit as st

from dogapp import config, predict, utils


@st.cache
def get_model_output(text) -> List:
    """"""
    # Get run components for prediction
    url, data, model = utils.get_run_components(text)
    # Get model output
    output = predict.predict(url=url, data=data, model=model)
    return output


def main():
    """"""
    # Application Pages
    page = st.sidebar.selectbox(
        "Choose a page", ["Dog Identification 🐶 App", "Other Apps.."]
    )

    if page == "Dog Identification 🐶 App":
        # Title
        st.title("Dog Identification 🐶 App")
        st.header("🚀 Try it out!")

        # Server warning
        st.info(
            """⚠️This app uses a **Small** web server. Too many request to it can **Overwhelm** it. In that case it will **Restart**. Please allow it to restart to make any further requests.✨"""
        )

        # Input box for image URL
        text = st.text_input(
            "Enter dog image url:",
            value="https://news.nationalgeographic.com/content/dam/news/2018/05/29/dog-baby-talk/01-dog-baby-talk-NationalGeographic_2283205.ngsversion.1527786004161.adapt.1900.1.jpg",
        )

        # Input image
        st.image(text, use_column_width=True)

        # Show results
        results = get_model_output(text)
        raw_text = results[0]["input_url"]
        preprocessed_text = results[0]["class"]
        st.write("**Input url**:", raw_text)
        st.write("**Predicted breed**:", preprocessed_text)

        # Show raw json
        show_json = st.checkbox("Show complete JSON output:", value=False)
        if show_json:
            st.json(results)

    elif page == "Other Apps..":

        st.title("Other Applications")

        st.write("More apps coming soon...")


main()
