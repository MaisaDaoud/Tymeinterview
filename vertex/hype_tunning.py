

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
print("Train machine type", TRAIN_COMPUTE)



aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

machine_spec = {"machine_type": TRAIN_COMPUTE, "accelerator_count": 0}

DISK_TYPE = "pd-ssd"  # [ pd-ssd, pd-standard]
DISK_SIZE = 100  # GB

disk_spec = {"boot_disk_type": DISK_TYPE, "boot_disk_size_gb": DISK_SIZE}

# Set path to save model
MODEL_DIR = "{}/aiplatform-custom-job".format(BUCKET_URI)

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
        "mse": "minimize",
    },
    parameter_spec={
            'learning_rate': hpt.DoubleParameterSpec(min=0.001, max=0.1, scale='log'),
            'max_depth': hpt.IntegerParameterSpec(min=4, max=8, scale='linear'),
            'n_estimators': hpt.DiscreteParameterSpec(values=[10,20, 30], scale='linear'),
           
    },
    search_algorithm=None,
    max_trial_count=4,
    parallel_trial_count=2,
)
   

hpt_job.run()

print('HT trails ', hpt_job.trials)

#best trail
# Initialize a tuple to identify the best configuration
best = (None, None, None, 0.0)
# Iterate through the trails and update the best configuration
for trial in hpt_job.trials:
    # Keep track of the best outcome
    if float(trial.final_measurement.metrics[0].value) > best[3]:
        try:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                float(trial.parameters[1].value),
                float(trial.final_measurement.metrics[0].value),
            )
        except:
            best = (
                trial.id,
                float(trial.parameters[0].value),
                None,
                float(trial.final_measurement.metrics[0].value),
            )

# print details of the best configuration
print('best ',best)


# Fetch the best model
#BEST_MODEL_DIR = MODEL_DIR + "/" + best[0] + "/model"

#gsutil ls {BEST_MODEL_DIR}
     






''' 
# Make folder for Python training script
#rm -rf vertex
mkdir vertex

# Add package information
touch vertex/README.md

setup_cfg="[egg_info]\n\ntag_build =\n\ntag_date = 0"
echo "$setup_cfg" > vertex/setup.cfg

setup_py="import setuptools\n\nsetuptools.setup(\n\n    install_requires=[\n\n        'cloudml-hypertune',\n\n    ],\n\n    packages=setuptools.find_packages())"
echo "$setup_py" > vertex/setup.py

pkg_info="Metadata-Version: 1.0\n\nName: Boston regression\n\nVersion: 0.0.0\n\nSummary: Demostration training script\n\nHome-page: www.maisa.com\n\nAuthor: Maisa\n\nAuthor-email: maysa_taheir@yahoo.com\n\nLicense: Public\n\nDescription: Demo\n\nPlatform: Vertex"
echo "$pkg_info" > vertex/PKG-INFO

# Make the training subfolder
mkdir vertex/trainer
touch vertex/trainer/__init__.py

''' 

'''
# zip package after local testing to copy it to the bucket
rm -f vertex.tar vertex.tar.gz
tar cvf trainer_boston.tar vertex
gzip trainer_boston.tar
gsutil cp trainer_boston.tar.gz gs://tym-maisa-doaud/trainer_boston.tar.gz
     

'''

