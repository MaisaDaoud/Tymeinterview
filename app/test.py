import pickle
import numpy as np
import os

def get_predictions(data):
    # try:
    print('********** ', os.getcwd() )
    with open(os.getcwd()+'/app/model/model.pkl', 'rb') as f:
        model = pickle.load(f)
        data = np.array(data)
    # except FileNotFoundError:
    #         model= None
    #         return 'No model found'
    # try:
    print('data ', data[:2])
    prediction = model.predict(data)
    # except:
    #         prediction = None
    #         'Error getting prediction. Unexpected data shape [], try [[]]'
        
    return prediction
  
    

