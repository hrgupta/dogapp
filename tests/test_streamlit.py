import os
from dogapp import dog  

def test_input_url():
    response = dog.get_model_output("https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg")
    assert response["data"]["prediction"][0]["input_url"] == "https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg"

def test_class():
    response = dog.get_model_output("https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg")
    with open(os.path.join(os.getcwd(), '.\dogapp\dog_names.txt')) as file:
        dog_names = file.read()
    dog_names = dog_names.split('\n')
    assert response["data"]["prediction"][0]["class"] in dog_names