import pandas as pd
import numpy as np
from functions import *
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

# Import Data

data = pd.read_csv('../input/data.csv')
test = pd.read_csv('../input/test.csv')

# Data Wrangling

submission = pd.DataFrame(test['id'])
test = clean(test)
test.drop('id', axis=1, inplace=True)

X, y = test_target(data)
X = clean(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Models

models = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(),
    "DecisionTreeRegresor": DecisionTreeRegressor(),
    "KNeighbors": KNeighborsRegressor(),
    "GradientBooster": GradientBoostingRegressor(),
}

# Train and check the models


for label, model in models.items():
    mae, mse, rmse, r2, y_pred, accuracies, score = train_model(
        X_train, y_train, model, X_test, y_test)
    print('')
    print('####### {} #######'.format(label))
    print('Score : %.4f' % score)
    print(accuracies)
    print('')
    print('MSE    : %0.2f ' % mse)
    print('MAE    : %0.2f ' % mae)
    print('RMSE   : %0.2f ' % rmse)
    print('R2     : %f ' % r2)

# Apply to the test

for name, model in models.items():
    print("applying {}...".format(name))
    model_final = model
    pred = predict_model(X, y, model_final, test)
    submission_model = submission
    submission_model["price"] = pred
    submission_model.to_csv(
        "./output/submission_{}.csv".format(name), index=False)
