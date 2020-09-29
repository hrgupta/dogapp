import os
from fastapi import FastAPI
from fastapi import Path
from fastapi.responses import RedirectResponse
from http import HTTPStatus
import json
from pydantic import BaseModel

from dogapp import config
from dogapp import utils
from dogapp import data
from dogapp import predict


app = FastAPI(
    title="Dog Identification 🐶 App",
    description="The app detects the breed of the dog 🐕 in the image given by the user. It also detects humans 💀 if the image contains a human.",
    version="1.0.0",
)

@utils.construct_response
@app.get("/")
async def _index():
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {}
    }
    config.logger.info(json.dumps(response, indent=2))
    return response


class PredictPayload(BaseModel):
    #experiment_id: str = 'latest'
    inputs: list = [{"url": ""}]


@utils.construct_response
@app.post("/predict")
async def _predict(payload: PredictPayload):

    url, data, model = predict.get_run_components(payload.inputs[0]['url'])

    prediction = predict.predict(url= url,data=data, model=model)

    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {"prediction": prediction}
    }
    #config.logger.info(json.dumps(response, indent=2))
    return response
