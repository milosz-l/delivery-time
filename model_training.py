
from random import Random
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
import pandas as pd
import numpy as np
from TimeDiffDataTransformer import TimeDiffDataTransformer
from TimeDiffConstants import DATE_FORMAT, PRICE_TRESHOLD, WEIGHT_TRESHOLD, NUM_OF_HOURS, SEED, COLS_TO_DROP_ALWAYS

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import time


def split_data(df, target_column="time_diff"):
    y = df["time_diff"].to_numpy()
    X = df.drop(columns="time_diff")
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def train_models(models_list, X_train, y_train):
    times = []
    for model in models_list:
        start = time.time()
        model.fit(X_train.values, y_train)
        end = time.time()
        times.append(end - start)
    return models_list, times


def create_df_with_predictions(models_list, X_test, y_test):
    y_pred_df = pd.DataFrame()
    y_pred_df["y_test"] = y_test
    for model in models_list:
        y_pred_df[f"{type(model).__name__} prediction"] = model.predict(X_test)
    return y_pred_df


def display_predictions(y_pred_df):
    display(y_pred_df.head())
    display(y_pred_df.info())
    display(y_pred_df.describe())


def print_scores(models_list, X_test, y_test, times):
    for model, time in zip(models_list, times):
        score = model.score(X_test, y_test)
        print(f"{type(model).__name__} score = {score}, training time = {time}s")


def print_percent_of_good_predictions(models_list, X_test, y_test, error=NUM_OF_HOURS*60*60):
    for model in models_list:
        predictions = model.predict(X_test)
        predictions_time_diff = np.abs(y_test - predictions)
        num_of_good_predictions = (predictions_time_diff < error).sum()
        percent_of_good_predictions = num_of_good_predictions / len(predictions_time_diff)
        print(f'number of good predictions for {type(model).__name__} = {num_of_good_predictions}/{len(predictions_time_diff)}')
        print(f'which is {percent_of_good_predictions * 100}% for +-{round(error/60/60)} hours\n')


warnings.filterwarnings('ignore')
# # preparing data
time_diff_data = TimeDiffDataTransformer()

time_diff_data.make_all_transformations()
df = time_diff_data.get_df()

X_train, X_test, y_train, y_test = split_data(df)


def getTrainedModels():
    # # models
    # models_list = [Ridge(alpha=0.1),
    #                Lasso(alpha=0.1),
    #                RidgeCV(),
    #                DecisionTreeRegressor(random_state=SEED),
    #                RandomForestRegressor(random_state=SEED)]
    models_list = [RidgeCV(),
                   RandomForestRegressor(random_state=SEED)]

    # # training model
    models_list, times = train_models(models_list, X_train, y_train)
    return models_list, times


models_list, times = getTrainedModels()

y_pred_df = create_df_with_predictions(models_list, X_test, y_test)
# display_predictions(y_pred_df)

# # printing results
print_scores(models_list, X_test, y_test, times)

print_percent_of_good_predictions(models_list, X_test, y_test)

print_percent_of_good_predictions(models_list, X_test, y_test, error=NUM_OF_HOURS/2*60*60)
