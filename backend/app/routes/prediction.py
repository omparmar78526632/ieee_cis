from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from app.schemas import PredictionInput

from app.services.prediction_service import PredictionService

from app.services.history_service import HistoryService

from app.database import get_db

router = APIRouter(

    prefix="/prediction",

    tags=["Prediction"]

)


@router.post("/predict")

def predict(

    data: PredictionInput,

    db: Session = Depends(get_db)

):

    result = PredictionService.predict(data)

    HistoryService.save_prediction(

        db,

        data.temperature,

        data.humidity,

        data.pressure,

        data.wind_speed,

        data.irradiation,

        result["predicted_power"],

        result["confidence"],

        result["model"]

    )

    return result


@router.get("/history")

def history(

    db: Session = Depends(get_db)

):

    return HistoryService.get_all_predictions(db)


@router.delete("/history/{id}")

def delete_prediction(

    id:int,

    db: Session = Depends(get_db)

):

    deleted=HistoryService.delete_prediction(db,id)

    return {

        "deleted":deleted

    }