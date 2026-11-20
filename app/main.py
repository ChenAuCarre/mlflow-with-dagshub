import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
# Trusted skops types were stored in the model at log time, so none needed here.
model = mlflow.sklearn.load_model("model")


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=64, max_length=64)  # flattened 8x8 digit


@app.post("/predict")
def predict(req: PredictRequest):
    return {"prediction": int(model.predict([req.features])[0])}
