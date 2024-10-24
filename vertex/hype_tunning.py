

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

import os
import json
import subprocess
from google.cloud import storage
from sklearn.model_selection import train_test_split
import pandas as pd

PROJECT_ID = "maisa-daoud"  # @param {type:"string"}
LOCATION = "us-west1"  # @param {type:"string"}
BUCKET_URI = f"gs://tym-maisa-doaud"


#create bucket command  if doesnt exists
# gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}


TRAIN_GPU, TRAIN_NGPU = (None, None)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)

TRAIN_VERSION = "xgboost-cpu.1-1"
DEPLOY_VERSION = "xgboost-cpu.1-1"

TRAIN_IMAGE = "{}-docker.pkg.dev/vertex-ai/training/{}:latest".format(
    LOCATION.split("-")[0], TRAIN_VERSION
)
DEPLOY_IMAGE = "{}-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(
    LOCATION.split("-")[0], DEPLOY_VERSION
)

TRAIN_COMPUTE = "n1-standard-4"

# Set path to save model
MODEL_DIR = "{}/aiplatform-custom-job/{}".format(BUCKET_URI, "model")

DISK_TYPE = "pd-ssd"  # [ pd-ssd, pd-standard]
DISK_SIZE = 100  # GB

def save_to_storage(file_name:str):
    storage_path = os.path.join(MODEL_DIR, file_name)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(file_name)



def split_save_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    #TODO propper datapreprocessing

    # raw_df[0].fillna(raw_df[0].mean(),inplace=True)
    # raw_df[1].fillna(raw_df[1].mean(),inplace=True)
    raw_df[10].fillna(raw_df[10].mean(),inplace=True)


    X, y = raw_df.iloc[:,0:10],  raw_df.iloc[:,10] #load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=35) #,stratify=y)

    pd.DataFrame(X_train).to_csv("x_train.csv",index=False)
    pd.DataFrame(y_train).to_csv("y_train.csv", index=False)

    pd.DataFrame(X_test).to_csv("x_test.csv",index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

    save_to_storage("x_train.csv")
    save_to_storage("y_train.csv")
    save_to_storage("x_test.csv")
    save_to_storage("y_test.csv")

  
def hyperune():
    split_save_data()

    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    machine_spec = {"machine_type": TRAIN_COMPUTE, "accelerator_count": 0}
    disk_spec = {"boot_disk_type": DISK_TYPE, "boot_disk_size_gb": DISK_SIZE}
    # # Set the command-line arguments
    CMDARGS = [
         #"--model-dir=" + MODEL_DIR,
         "--training-url="+os.path.join(MODEL_DIR, "x_train.csv"),
         "--labels-url="+os.path.join(MODEL_DIR, "y_train.csv"),
         "--testing-url="+os.path.join(MODEL_DIR, "x_test.csv"),
         "--testing-labels-url="+os.path.join(MODEL_DIR, "y_test.csv")
    ]

    # Set the worker pool specs
    worker_pool_spec = [
        {
            "replica_count": 1,
            "machine_spec": machine_spec,
            "disk_spec": disk_spec,
            "python_package_spec": {
                "executor_image_uri": TRAIN_IMAGE,
                "package_uris": [BUCKET_URI + "/trainer_boston.tar.gz"],
                "python_module": "trainer.task",
                 "args": CMDARGS,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="boston",
        worker_pool_specs=worker_pool_spec,
        base_output_dir=MODEL_DIR,
        staging_bucket=MODEL_DIR,
    )


    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="boston",
        custom_job=job,
        metric_spec={
        "mse": "maximize",
        },
        parameter_spec={
                'learning_rate': hpt.DoubleParameterSpec(min=0.09, max=0.2, scale='liear'),
                'max_depth': hpt.IntegerParameterSpec(min=4, max=6, scale='linear'),
                'n_estimators': hpt.DiscreteParameterSpec(values=[10,15,20,25,30], scale='linear'),
            
        },
        search_algorithm=None,
        max_trial_count=2,
        parallel_trial_count=2,
    )
    

    hpt_job.run()

    print('HT trails ', hpt_job.trials)

    #best trail
    # Initialize a tuple to identify the best configuration
    best = (None, None, None, -100.0)
    # Iterate through the trails and update the best configuration
    for trial in hpt_job.trials:
        # Keep track of the best outcome
        if float(trial.final_measurement.metrics[0].value) > best[3]: #>
            try:
                best = (
                    trial.id,
                    float(trial.parameters[0].value),
                    float(trial.parameters[1].value),
                    float(trial.parameters[2].value),
                    float(trial.final_measurement.metrics[0].value),
                )
            except:
                best = (
                    trial.id,
                    float(trial.parameters[0].value),
                    None,
                   
                )

    # print details of the best configuration
    print('best ',best)

    # Fetch the best model
    BEST_MODEL_DIR = MODEL_DIR + "/" + best[0] + "/model"
    print('BEST_MODEL_DIR ', BEST_MODEL_DIR)


    
    #write best results to file
    with open("best_parameters.json", 'w') as outfile:
        json.dump({ "model_id":best[0],"learning_rate":best[1],"max_depth":best[2],"n_estimators":best[3],"model_dir":BEST_MODEL_DIR}, outfile)
    
    #copy best model to app/model/model.pkl
    subprocess.run(["gsutil","-m","cp","-r",BEST_MODEL_DIR , "../app/."])
    # #gsutil ls {BEST_MODEL_DIR}

if __name__=='__main__':
    hyperune() 


