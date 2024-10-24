
import os
import pandas as pd
import xgboost as xgb
import hypertune
import argparse
import logging
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from google.cloud import storage
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='defaults is the staging bucket location')
parser.add_argument("--learning_rate", dest="learning_rate",
                    default=0.1, type=float, help="learning rate for training")

parser.add_argument("--n_estimators", dest="n_estimators",
                    default=20, type=int, help="number ofestimators")

parser.add_argument("--max_depth", dest="max_depth",
                    default=5, type=int, help="max depth")                        

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def get_data():
    logging.info("Downloading training data and labels")
    # gsutil outputs everything to stderr. Hence, the need to divert it to stdout.
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    #TODO propper datapreprocessing

    raw_df[0].fillna(raw_df[0].mean(),inplace=True)
    raw_df[1].fillna(raw_df[1].mean(),inplace=True)
    raw_df[10].fillna(raw_df[10].mean(),inplace=True)


    X, y = raw_df.iloc[:,0:10],  raw_df.iloc[:,10] #load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=35) #,stratify=y)

    # Load data into DMatrix object
    #dtrain = xgb.DMatrix(X_train, label=y_train)
    return X_train,y_train, X_test, y_test

def train_model(X_train,y_train):
    logging.info("Start training ...")
    # Train XGBoost model
    xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", # reg:linear
                          learning_rate=args.learning_rate,
                          n_estimators=args.n_estimators,
                          n_jobs=-1,
                          colsample_bytree=0.3,
                          max_depth=args.max_depth,
                          alpha=10,
                          random_state=42)
    xgb_reg.fit(X_train, y_train)
    with open('model.pkl','wb') as f:
        pickle.dump(xgb_reg,f)
    logging.info("Training completed")
    
    return xgb_reg

def evaluate_model(model, test_data, test_labels):
   
    y_hat = model.predict(test_data)

    # evaluate predictions
    mse = - mean_squared_error(test_labels, y_hat)
    logging.info(f"Evaluation completed with model mse: {mse}")

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mse',
        metric_value=mse
    )
    return mse


X_train,y_train,X_test,y_test  = get_data()
pd.DataFrame(X_test) .to_csv("x_test.csv",index=False)
pd.DataFrame(y_test) .to_csv("y_test.csv", index=False)
model = train_model(X_train,y_train)
mse = evaluate_model(model, X_test, y_test)


# GCSFuse conversion
# gs_prefix = 'gs://'
# gcsfuse_prefix = '/gcs/'
# if args.model_dir.startswith(gs_prefix):
#     args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
#     dirpath = os.path.split(args.model_dir)[0]
#     if not os.path.isdir(dirpath):
#         os.makedirs(dirpath)

# # Export the classifier to a file
# gcs_model_path = os.path.join(args.model_dir, 'model.pkl')
# logging.info("Saving model artifacts to {}". format(gcs_model_path))
# model.save_model(gcs_model_path)

# logging.info("Saving metrics to {}/mse.json". format(args.model_dir))
# gcs_metrics_path = os.path.join(args.model_dir, 'metrics.json')
# with open(gcs_metrics_path, "w") as f:
#     f.write(f"{'mse: {mse}'}")
artifact_filename = 'model.pkl'

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
with open(local_path, 'wb') as model_file:
  pickle.dump(model, model_file)

# Upload model artifact to Cloud Storage

storage_path = os.path.join(args.model_dir, artifact_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)