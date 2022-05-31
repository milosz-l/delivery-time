from flask import Flask, request
from model_training import *
import pandas as pd

app = Flask(__name__)

models = getTrainedModels()

from TimeDiffDataTransformer import *
time_diff_data = TimeDiffDataTransformer()

time_diff_data.make_all_transformations()
df = time_diff_data.get_df()

print(models)

def preprocessData(data):
    df = pd.DataFrame(data)
    df["purchase_datetime_delta"] = 0
    df["time_diff"] = 0
    df["delivery_timestamp"] = 0
    return df

@app.route('/', methods=['POST'])
def result():
    print("NEW POST REQUEST")
    data = request.json
    print("DATA", data)  # json (if content-type of application/json is sent with the request)

    df = preprocessData(data)
    print(df.keys())

    # prediction_base_model = models[0].predict(data)
    # print(prediction_base_model)

    return "OK"