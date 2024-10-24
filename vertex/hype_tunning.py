

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

import os

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
MODEL_DIR = "{}/aiplatform-custom-job".format(BUCKET_URI)

DISK_TYPE = "pd-ssd"  # [ pd-ssd, pd-standard]
DISK_SIZE = 100  # GB

def hyperune():
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    machine_spec = {"machine_type": TRAIN_COMPUTE, "accelerator_count": 0}
    disk_spec = {"boot_disk_type": DISK_TYPE, "boot_disk_size_gb": DISK_SIZE}
    # # Set the command-line arguments
    # CMDARGS = [
    #     "--dataset-data-url=" + DATASET_DIR + "/iris_data.csv",
    #     "--dataset-labels-url=" + DATASET_DIR + "/iris_target.csv",
    # ]

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
                # "args": CMDARGS,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="boston",
        worker_pool_specs=worker_pool_spec,
        base_output_dir=MODEL_DIR,
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
        max_trial_count=6,
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
                    float(trial.final_measurement.metrics[0].value),
                    trial.final_measurement.metrics[0].value,
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

    # #gsutil ls {BEST_MODEL_DIR}

if __name__=='__main__':
    hyperune() 


