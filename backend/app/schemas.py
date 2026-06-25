from pydantic import BaseModel
from datetime import datetime


# ----------------------------
# Prediction Request Schema
# ----------------------------

class PredictionRequest(BaseModel):

    temperature: float

    humidity: float

    pressure: float

    wind_speed: float

    irradiation: float


# ----------------------------
# Prediction Response Schema
# ----------------------------

class PredictionResponse(BaseModel):

    predicted_power: float

    model_name: str

    message: str


# ----------------------------
# Prediction History Schema
# ----------------------------

class PredictionHistory(BaseModel):

    datetime: datetime

    temperature: float

    humidity: float

    pressure: float

    wind_speed: float

    irradiation: float

    predicted_power: float

    model_name: str

    class Config:
        from_attributes = True


# ----------------------------
# Dataset Information
# ----------------------------

class DatasetInfo(BaseModel):

    rows: int

    columns: int

    missing_values: int

# ----------------------------
# Prediction Input
# ----------------------------

class PredictionInput(BaseModel):

    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    irradiation: float


# ----------------------------
# Prediction Output
# ----------------------------

class PredictionOutput(BaseModel):

    predicted_power: float

    confidence: float

    model: str

    status: str

class DecisionInput(BaseModel):

    temperature: float

    humidity: float

    pressure: float

    wind_speed: float

    irradiation: float

    predicted_power: float