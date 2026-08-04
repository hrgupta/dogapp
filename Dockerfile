FROM python:3.14-slim

# Modernized stack: Keras 3 on the PyTorch backend.
# (TensorFlow has no Python 3.14 wheels, so tf.keras is gone.)
ENV KERAS_BACKEND=torch

# Install dependencies (modernized for Python 3.14 — see requirements.txt)
RUN apt-get update && apt-get -y install supervisor
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy
COPY . .

# Export ports
EXPOSE 5000
EXPOSE 8501

# Start FastAPI + Streamlit via supervisord
CMD ["/usr/bin/supervisord", "-c", "/supervisor/service_script.conf"]
