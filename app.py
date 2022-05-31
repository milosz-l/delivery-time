from flask import Flask, request
from model_training import *

app = Flask(__name__)

models = getTrainedModels()
print(models)

@app.route('/', methods=['POST'])
def result():
    print("NEW POST REQUEST")
    data = request.json
    print("DATA", data)  # json (if content-type of application/json is sent with the request)


    return "OK"