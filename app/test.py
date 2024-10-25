import pickle
import numpy as np
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'model/model.pkl')


def get_predictions(data):
    # try:
    with open(filename, 'rb') as f:  #'app/model/model.pkl'
        model = pickle.load(f)
        data = np.array(data)
    # except FileNotFoundError:
    #         model= None
    #         return 'No model found'
    # try:
    prediction = model.predict(data)
    # except:
    #         prediction = None
    #         'Error getting prediction. Unexpected data shape [], try [[]]'
        
    return prediction
  
    

