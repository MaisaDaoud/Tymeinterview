

import pandas as pd
from app.test import get_predictions
from sklearn.metrics import mean_squared_error
import json 

data = pd.read_csv('x_test.csv')
y = pd.read_csv('y_test.csv')

y_hat = get_predictions(data)

mse = mean_squared_error(y, y_hat)
    
# Write results to file
with open("test_score.json", 'w') as outfile:
        json.dump({ "mse":mse}, outfile)
