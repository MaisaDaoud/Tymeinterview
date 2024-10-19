import pickle
import logging
import numpy as np

def get_predictions(data):
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
            model= None
            logging.warning('No model found')
  
    data = np.array(data)
    try:
        prediction = model.predict(data)
    except:
         'Error getting prediction. Unexpected data shape [], try [[]]'
    return prediction

