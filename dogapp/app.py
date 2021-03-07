import os
import sys

from numpy.lib import utils

sys.path.append(".")
import json
from http import HTTPStatus

from fastapi import FastAPI, Path
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from dogapp import config, predict, models, utils

app = FastAPI(
    title="Dog Identification App",
    description="",
    version="1.0.0",
)


@utils.construct_response
@app.get("/")
async def _index():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


class PredictPayload(BaseModel):
    # experiment_id: str = "latest"
    # inputs: list = [{"url": ""}]
    urls: list = [""]


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):

    # Get run components for prediction
    model = models.DogCNN()
    model.summary(input_shape=(7, 7, 2048))  # build it
    model_path = os.path.join(os.getcwd(), "embeddings/weights.best.Xception.hdf5")
    model.load_weights(model_path)
    prediction = []
    for url in payload.urls:
        try:
            data = utils.loadImage(url)
            prediction.append(predict.predict(url=url, data=data, model=model)[0])
            response = {
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
                "data": {"prediction": prediction},
            }
            config.logging.getLogger('infologger').info(json.dumps(response, indent=2))
        except Exception as pe:
            config.logging.getLogger('errorlogger').error("Error making predictions", exc_info=pe)
    return response
