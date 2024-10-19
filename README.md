# Tymeinterview

# local testing 
## set up the conda env
conda create --name tym python=3.9
conda activate tym
pip install -r requirements.txt

### testt the server locally 
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


# To build container image 
docker build --platform linux/amd64 -t boston-predict 
docker run -p 8000:8000 boston-predict
docker tag boston-predict:latest  mtdd1/tyminterview:latest
docker login
docker push mtdd1/tyminterview:latest 
