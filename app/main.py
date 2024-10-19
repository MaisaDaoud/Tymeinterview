
import numpy as np
from fastapi import FastAPI 
from pydantic import BaseModel
import uvicorn
from test import get_predictions

app = FastAPI()

class PredictRequest(BaseModel):
    data: list


@app.get("/")
def index():
    return {'Hello':'world'}

@app.post("/predict")
async def predict(request:PredictRequest):
    data = request.data
    prediction = get_predictions(data)
    return  {'message': f'prediction is {prediction}'}

    
if __name__== "__main__":
    port = 8000
    print(f'running fastapi on {port}')
    uvicorn.run("main:app", host='0.0.0.0', port=port)

    

#print(predict_price(np.array([[0.00632,18.00,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0]])))
