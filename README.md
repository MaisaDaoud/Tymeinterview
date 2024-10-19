# Tymeinterview

# local testing setting up the env
conda create --name tym python=3.9
conda activate tym

# testthe server locally 
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


# Docker 
docker build --platform linux/amd64 -t boston-predict 
docker run -p 8000:8000 boston-predict
