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


def preprocessData(df):
    # 1.
        df["delivery_timestamp"] = df["purchase_timestamp"].str.split('.', expand=True)[0]

        # 2.
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], format=DATE_FORMAT)
        df["delivery_timestamp"] = pd.to_datetime(df["delivery_timestamp"], format=DATE_FORMAT)

        # 3.
        df["time_diff"] = df["delivery_timestamp"] - df["purchase_timestamp"]

        # 4.
        # df = df[df["time_diff"].notna()]

        # 5.
        # time diff as duration in seconds
        df["time_diff"] = df["time_diff"].apply(datetime.timedelta.total_seconds)

        # drop rows where event_type is not equal "BUY_PRODUCT"
        df = df[df["event_type"] == "BUY_PRODUCT"]

        # making sure, that timestamp == purchase_timestamp
        # num_of_rows_before = df.shape[0]
        # df = df[df["timestamp"] == df["purchase_timestamp"]]
        # num_of_rows_after = df.shape[0]
        # assert(num_of_rows_before == num_of_rows_after)

        # now we can drop timestamp column, as it is redundant
        # df = df.drop(columns="timestamp")

        # df = df.merge(df, on="user_id", how="left")
        # df = df.merge(df, on="product_id", how="left")

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
    data = request.json

    df = pd.DataFrame(data)
    df = preprocessData(df)
    print("AFTER PREPROCESS", df)

    new_data = TimeDiffDataTransformer(df)
    test_df = new_data.get_df()
    print("TEST DF", test_df)

    new_data.make_all_transformations()
    df_new = new_data.get_df()
    df_compare = df_new.iloc[:1]
    df_new = df_new.iloc[-1:]

    print("DF_NEW", df_new)
    print("DF_NEW2", df)

    print("DF_COMPARE", df_compare)

    #porownanie kluczy
    for key in df_compare.keys():
        if key not in df_new.keys():
            print ("NO KEY",key)

    #porownanie typów
    # for key in df_compare.keys():
    #     # if type(df_compare[key][0]) != type(df_new[key][0]):
    #     print ("DIFF", key, type(df_compare[key][0]), df_compare[key][0], type(df_new[key][0]), df_new[key][0])

    y_test = df_new["time_diff"].to_numpy()
    x_test = df_new.drop(columns="time_diff")

    print("YTEST", y_test)
    print("XTEST", x_test)

    y_pred_df = create_df_with_predictions(models, x_test, y_test)

    print("Predictions", y_pred_df)

    # return y_pred_df.to_dict()
    return "OK"