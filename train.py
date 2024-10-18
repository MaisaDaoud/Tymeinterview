import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import xgboost as xgb
import pickle




# Load Datasets
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

#TODO propper datapreprocessing

raw_df[0].fillna(raw_df[0].mean(),inplace=True)
raw_df[1].fillna(raw_df[1].mean(),inplace=True)
# raw_df[2].fillna(raw_df[2].mean(),inplace=True)
# raw_df[3].fillna(raw_df[3].mean(),inplace=True)
# raw_df[4].fillna(raw_df[4].mean(),inplace=True)
# raw_df[5].fillna(raw_df[5].mean(),inplace=True)
raw_df[10].fillna(raw_df[10].mean(),inplace=True)


X, y = raw_df.iloc[:,0:10],  raw_df.iloc[:,10] #load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=34) #,stratify=y)


xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", # reg:linear
                          learning_rate=0.1,
                          n_estimators=10,
                          n_jobs=-1,
                          colsample_bytree=0.3,
                          max_depth=5,
                          alpha=10,
                          random_state=42)
xgb_reg.fit(X_train, y_train)
# xgb_reg.save_model("model/clf.json")
# save
with open('model/model.pkl','wb') as f:
    pickle.dump(xgb_reg,f)
y_pred = xgb_reg.predict(X_test)
print(y_pred)
#TODO use evaluation metrics for validation

