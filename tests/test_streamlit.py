import os
from unittest import mock

from dogapp import dog

URL = "https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg"


def _fake_response(*args, **kwargs):
    """Return a canned /predict response without needing a live server."""

    class R:
        text = (
            '{"message":"OK","status-code":200,"data":{"prediction":'
            '[{"input_url":"' + URL + '","class":"Anatolian Shepherd Dog"}]}}'
        )

    return R()


@mock.patch("requests.post", side_effect=_fake_response)
def test_input_url(mock_post):
    response = dog.get_model_output(URL)
    assert response["data"]["prediction"][0]["input_url"] == URL


@mock.patch("requests.post", side_effect=_fake_response)
def test_class(mock_post):
    response = dog.get_model_output(URL)
    with open(os.path.join(os.getcwd(), "./dogapp/dog_names.txt")) as file:
        dog_names = file.read()
    dog_names = dog_names.split("\n")
    assert response["data"]["prediction"][0]["class"] in dog_names
