# Dog Identification 🐶 App

Web app that detects dog breeds from image URLs — FastAPI backend + Streamlit UI, plus a CLI for one-off predictions.

> **Modernized in 2026:** originally built on TensorFlow 2.2 / Python 3.7 (2020), which is no longer installable. Now runs on **Python 3.14 with Keras 3 on the PyTorch backend** — TensorFlow publishes no Python 3.14 wheels.

## Stack

- **Python 3.14** · **Keras 3 (PyTorch backend)** · FastAPI · Streamlit · Docker
- Model: Xception bottleneck features → 133-breed classification head (`embeddings/weights.best.Xception.hdf5`)

## Set up

Requires [uv](https://docs.astral.sh/uv/) (or any Python 3.14 interpreter):

```bash
uv venv --python 3.14 venv
uv pip install -r requirements.txt
```

> Run app commands with `KERAS_BACKEND=torch` in the environment. The first inference downloads the Xception ImageNet weights (~85 MB).

## Inference via CLI

```bash
KERAS_BACKEND=torch venv/bin/python dogapp/predict.py --url <image-url>
```

## Run the API

```bash
KERAS_BACKEND=torch venv/bin/uvicorn dogapp.app:app --host 127.0.0.1 --port 5000
→ http://127.0.0.1:5000/docs
```

> **macOS note:** use `127.0.0.1`, not `localhost` — macOS AirPlay Receiver squats port 5000 and answers `localhost`/`::1` requests with a denial.

## Run the Streamlit UI

```bash
venv/bin/streamlit run dogapp/dog.py
→ http://localhost:8501
```

## Tests

```bash
KERAS_BACKEND=torch venv/bin/python -m pytest
```

## Docker

```bash
docker build -t dogapp:latest .
docker run -d -p 8000:5000 -p 8502:8501 --name dogapp dogapp:latest
→ API: http://127.0.0.1:8000 · UI: http://127.0.0.1:8502
```

## Directory structure

```
dogapp/          app endpoints (app.py), config, model (models.py),
                 inference (predict.py), utilities (utils.py),
                 dog_names.txt, embeddings/
tests/           API + Streamlit tests
.github/workflows/  CI (Python 3.14 · flake8 + pytest)
```

## CI

[![DogApp CI](https://github.com/hrgupta/dogapp/actions/workflows/python-app.yml/badge.svg)](https://github.com/hrgupta/dogapp/actions/workflows/python-app.yml)
