# https://blog.finxter.com/deploying-a-machine-learning-model-in-fastapi/

import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()


pickle_in = open("demo_mlr.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
   return {"message": "Hello World"}

@app.post("/predict")
def predict(x1: float, x2: float):
   x2 = np.log(x2)
   data = np.array([[x1, x2]])
   prediction = model.predict(data)
   return {
       'prediction': prediction[0],
   }


if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)
