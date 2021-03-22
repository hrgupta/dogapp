FROM python:3.7

# Install dependencies
# Do this first for caching
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy
COPY . .

# Export ports
EXPOSE 5000

# Start app
CMD ["uvicorn", "dogapp.app:app", "--host", "0.0.0.0", "--port", "5000"]