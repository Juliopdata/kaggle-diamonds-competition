import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error
import math

# Split the dataframe


def test_target(df):
    y = df['price']
    X = df.drop(columns=['price'])
    return X, y

# Cleaning Function


def clean(df):
    features = {'cut': {'Fair': 1, 'Good': 2, 'Ideal': 3, 'Premium': 4, 'Very Good': 5}, 'color': {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}, 'clarity': {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4,
                                                                                                                                                                        'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}}
    for col, eq in features.items():
        for cat, val in eq.items():
            df[col] = df[col].apply(lambda x: val if x == cat else x)
    df.drop('x', axis=1, inplace=True)
    df.drop('z', axis=1, inplace=True)
    return df

# Apply the prediction model


def predict_model(X, y, model, test):
    model.fit(X, y)
    pred = model.predict(test)
    return pred

# Train the model


def train_model(X_train, y_train, model, X_test, y_test):
    clf = model
    clf.fit(X_train, y_train)
    accuracies = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=5, verbose=1)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    score = clf.score(X_test, y_test)
    return (mae, mse, rmse, r2, y_pred, accuracies, score)
