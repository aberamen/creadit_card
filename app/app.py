from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("../model/saved_models.joblib")

app = FastAPI()

# Define the input data structure (the features expected from the user)
class TransactionData(BaseModel):
    amount: float
    transaction_time: str
    # Add other features based on your model input

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: TransactionData):
    # Convert incoming data to a NumPy array (or whatever format the model expects)
    features = np.array([[data.amount]])  # Adjust based on your features
    prediction = model.predict(features)
    
    # Return the prediction (for example, 1 for high risk, 0 for low risk)
    return {"prediction": int(prediction[0])}
