import pickle
import numpy as np
import xgboost as xgb

def predict_price(X):
    # load
    with open('model/model.pkl', 'rb') as f:
        clf2 = pickle.load(f)
    return clf2.predict(X) #[0:1])

print(predict_price(np.array([[0.00632,18.00,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0]])))




