# Dog Identification 🐶 App

🚀 This project was created using the [e2e-ml-app](https://github.com/madewithml/e2e-ml-app-tensorflow) cookiecutter template. Check it out to start creating your own ML applications.

## Set up

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt [update requirements.txt as needed]
```

## Inference via scripts

```bash
python dogapp/predict.py --url <image-url>
```

## Endpoints

```bash
uvicorn dogapp.app:app --host 0.0.0.0 --port 5000 --reload
→ http://localhost:5000/docs
```
## Docker

1. Build image

```bash
docker build -t dog-who:latest -f Dockerfile .
```

2. Run container

```bash
docker run -d -p 5000:5000 -p 6006:6006 --name dog-who dog-who:latest
```

## Directory structure

```
dogapp/
├── experiments/                        - experiment directories
├── logs/                               - directory of log files
|   ├── errors/                           - error log
|   └── info/                             - info log
├── tests/                              - unit tests
├── dogapp/
|   ├── app.py                            - app endpoints
|   ├── config.py                         - configuration
|   ├── data.py                           - data processing
|   ├── models.py                         - model architectures
|   ├── predict.py                        - inference script
|   ├── train.py                          - training script
|   ├── utils.py                          - load embeddings
|   ├── dog_names.txt                     - contains dog names
|   └── weights.bext.Xception.hdf5        - learned weights of Xception model
├── .dockerignore                       - files to ignore on docker
├── .gitignore                          - files to ignore on git
├── .slugignore                         - files to ignore on slug
├── CODE_OF_CONDUCT.md                  - code of conduct
├── CODEOWNERS                          - code owner assignments
├── config.py                           - configuration
├── CONTRIBUTING.md                     - contributing guidelines
├── Dockerfile                          - dockerfile to containerize app
├── LICENSE                             - license description
├── logging.json                        - logger configuration
├── README.md                           - this README
├── setup.sh                            - setup file
└── requirements.txt                    - requirements
```

## Helpful docker commands

• Build image

```
docker build -t dog-who:latest -f Dockerfile .
```

• Run container if using `CMD ["python", "app.py"]` or `ENTRYPOINT [ "/bin/sh", "entrypoint.sh"]`

```
docker run -p 5000:5000 --name dog-who dog-who:latest
```

• Get inside container if using `CMD ["/bin/bash"]`

```
docker run -p 5000:5000 -it dog-who /bin/bash
```

• Other flags

```
-d: detached
-ti: interative terminal
```

• Clean up

```
docker stop $(docker ps -a -q)     # stop all containers
docker rm $(docker ps -a -q)       # remove all containers
docker rmi $(docker images -a -q)  # remove all images
```
