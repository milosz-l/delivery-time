from flask import Flask, request
from model_training import *
import pandas as pd
from TimeDiffDataTransformer import *
from TimeDiffConstants import *


app = Flask(__name__)

time_diff_data = TimeDiffDataTransformer()

time_diff_data.make_all_transformations()
df_trained = time_diff_data.get_df()

models = getTrainedModels()

@app.route('/', methods=['POST'])
def handleRequest():
    data = request.json

    df = pd.DataFrame(data)

    new_data = TimeDiffDataTransformer()
    new_data.make_all_transformations(df)
    df_new = time_diff_data.get_df()
    df_new = df_new.iloc[-1:]

    y_test = df_new["time_diff"].to_numpy()
    x_test = df_new.drop(columns="time_diff")

    y_pred_df = create_df_with_predictions(models_list, x_test, y_test)

    print("Predictions", y_pred_df)

    return y_pred_df.to_dict()