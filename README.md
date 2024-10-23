# Tymeinterview

## local testing 
## set up the conda env
conda create --name tym python=3.9
conda activate tym
pip install -r requirements.txt





## Hyperparameter tunning with VertexAI
# Prepare the trainer packege
```
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
cp task_copy  vertex/trainer/task.py 

#test task.py locally
```
# Push trainer package to the bucket
```
# create bucket if not exists
gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}
# zip package after local testing to copy it to the bucket
rm -f trainer_boston.tar trainer_boston.tar.gz
tar cvf trainer_boston.tar -C vertex .
gzip trainer_boston.tar
gsutil cp trainer_boston.tar.gz gs://tym-maisa-doaud/trainer_boston.tar.gz
```


## test the server locally 
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


## To build  the container image locally
```
docker build --platform linux/amd64 -t boston-predict .
docker run -p 8000:8000 boston-predict
docker tag boston-predict:latest  mtdd1/tyminterview:latest
docker login
docker push mtdd1/tyminterview:latest 
```


## K8s
This repo is tested with local k8s, to replicate the deployment steps
1.  make sure that minikube and kubctl are installed
```
minikube start --nodes=2
kubectl apply -f deployment.yml
kubectl get deploy
kubectl get pods
kubectl get svc
minikube service myapp
```


## challenges I faced
it took me some time to figure out how to solve `dvc command not found bug`, not many available onine solutions. 
Turns out the I should include the version as per the documentation suggests
