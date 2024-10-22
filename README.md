# Tymeinterview

## local testing 
## set up the conda env
conda create --name tym python=3.9
conda activate tym
pip install -r requirements.txt

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
