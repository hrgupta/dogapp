FROM python:3.7-buster

# Install dependencies
# Do this first for caching
RUN apt-get update
RUN apt-get -y install supervisor
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy
COPY . .

# Export ports
EXPOSE 8501
EXPOSE 8502
EXPOSE 5000

# Start app
#CMD ["uvicorn", "dogapp.app:app", "--host", "0.0.0.0", "--port", "5000"]
#CMD ["sh", "-c","streamlit run dogapp/dog.py"]
CMD /usr/bin/supervisord -c /supervisor/service_script.conf