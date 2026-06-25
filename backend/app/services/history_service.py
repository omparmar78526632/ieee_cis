from sqlalchemy.orm import Session

from app.models import PredictionHistory


class HistoryService:


    @staticmethod
    def save_prediction(

        db: Session,

        temperature,

        humidity,

        pressure,

        wind_speed,

        irradiation,

        prediction,

        confidence,

        model

    ):

        history = PredictionHistory(

            temperature=temperature,

            humidity=humidity,

            pressure=pressure,

            wind_speed=wind_speed,

            irradiation=irradiation,

            predicted_power=prediction,

            confidence=confidence,

            model_name=model

        )

        db.add(history)

        db.commit()

        db.refresh(history)

        return history


    @staticmethod
    def get_all_predictions(db: Session):

        return db.query(PredictionHistory).all()


    @staticmethod
    def delete_prediction(db: Session,id):

        prediction=db.query(PredictionHistory).filter(

            PredictionHistory.id==id

        ).first()

        if prediction:

            db.delete(prediction)

            db.commit()

            return True

        return False