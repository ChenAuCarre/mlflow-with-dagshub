import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
# Same trusted type as when logging (MLflow 3 skops serialization).
model = mlflow.sklearn.load_model("model", skops_trusted_types=["sklearn.tree._tree.Tree"])


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=64, max_length=64)  # flattened 8x8 digit


@app.post("/predict")
def predict(req: PredictRequest):
    return {"prediction": int(model.predict([req.features])[0])}
