from itertools import count
from flask import Flask, request
from model_training import *
import pandas as pd
from TimeDiffDataTransformer import *
from TimeDiffConstants import *


app = Flask(__name__)

models, times = getTrainedModels()

counter = 0

def preprocessData(df):

    df["delivery_timestamp"] = df["purchase_timestamp"].str.split('.', expand=True)[0]

    df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], format=DATE_FORMAT)
    df["delivery_timestamp"] = pd.to_datetime(df["delivery_timestamp"], format=DATE_FORMAT)

    df["time_diff"] = df["delivery_timestamp"] - df["purchase_timestamp"]

    df["time_diff"] = df["time_diff"].apply(datetime.timedelta.total_seconds)

    # drop rows where event_type is not equal "BUY_PRODUCT"
    df = df[df["event_type"] == "BUY_PRODUCT"]

    # rejecting outliers for given PRICE_TRESHOLD
    df = df[df["price"] <= PRICE_TRESHOLD]

    # rejecting outliers for given WEIGHT_TRESHOLD
    df = df[df["weight_kg"] <= WEIGHT_TRESHOLD]

    # deleting rows with prices below 0
    df = df[df["price"] >= 0]

    # deleting rows with time_diff below 0
    df = df[df["time_diff"] >= 0]

    # adding column with day of week
    df['day_of_week'] = df['purchase_timestamp'].dt.dayofweek

    # adding city_and_street interaction column
    df['city_and_street'] = df['city'] + ' ' + df['street']

    # adding continuous variable from purchase_timestamp (days from the first date)
    df['purchase_datetime_delta'] = (df['purchase_timestamp'] - df['purchase_timestamp'].min()) / np.timedelta64(1, 'D')

    return df


@app.route('/', methods=['POST'])
def handleRequest():
    # data from request
    data = request.json

    # data frame of a counter
    global counter
    df_counter = pd.DataFrame({"counter": [counter]})

    # preprocessing data
    df = pd.DataFrame(data)

    # saving request to logs
    df_counter.join(df).to_csv('logs/requests.csv', index=False, mode='a')

    df = preprocessData(df)
    purchase_timestamp = df["purchase_timestamp"]

    # transformations
    new_data = TimeDiffDataTransformer(df)
    new_data.make_all_transformations()
    df_new = new_data.get_df()
    # getting last row of transformed data
    df_new = df_new.iloc[-1:]

    # data frames to test
    y_test = df_new["time_diff"].to_numpy()
    x_test = df_new.drop(columns="time_diff")

    # predictions
    y_pred_df = create_df_with_predictions(models, x_test, y_test)
    print("Predictions", y_pred_df)

    # calculating final prediction - predicted date of delivery
    # random forest
    delivery_timestamp_random_forest = purchase_timestamp + datetime.timedelta(0, y_pred_df["RandomForestRegressor prediction"][0])
    print("Predicted delivery (Random Forest): ", delivery_timestamp_random_forest[0])

    # ridge
    delivery_timestamp_ridge = purchase_timestamp + datetime.timedelta(0, y_pred_df["RidgeCV prediction"][0])
    print("Predicted delivery (Ridge CV): ", delivery_timestamp_ridge[0])

    # saving prediction to logs
    df_random_forest = pd.DataFrame(delivery_timestamp_random_forest)
    df_ridge = pd.DataFrame(delivery_timestamp_ridge)
    df_counter.join(df_random_forest).to_csv('logs/RandomForestPredictions.csv', index=False, mode='a')
    df_counter.join(df_ridge).to_csv('logs/RidgeCVPredictions.csv', index=False, mode='a')
    
    # increasing counter
    counter += 1
    
    return {
        "RandomForestRegressor prediction": delivery_timestamp_random_forest[0],
        "Ridge prediction": delivery_timestamp_ridge[0]
    }
