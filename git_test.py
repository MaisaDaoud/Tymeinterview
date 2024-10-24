

import pandas as pd
from app.test import get_predictions
from sklearn.metrics import mean_squared_error
import json 
import subprocess
import sys
import os

# BUCKET_URI = f"gs://tym-maisa-doaud"
# MODEL_DIR = "{}/aiplatform-custom-job/{}".format(BUCKET_URI, "model")
# subprocess.check_call(['gsutil', 'cp', os.path.join(MODEL_DIR, "x_test.csv"), 'x_test.csv'], stderr=sys.stdout)
# subprocess.check_call(['gsutil', 'cp', os.path.join(MODEL_DIR, "y_test.csv"), 'y_test.csv'], stderr=sys.stdout)

data = pd.read_csv('x_test.csv')
y = pd.read_csv('y_test.csv')

y_hat = get_predictions(data)

mse = mean_squared_error(y, y_hat)

# get file
try:
    test_score = json.load(open('test_score.json'))
except FileNotFoundError:
    test_score = {} 

test_score["mse"] = "mse"  

# Write results to file
with open("test_score.json", 'w') as outfile:
        print('IM HERE ********')
        json.dump({ "mse":mse}, outfile)
