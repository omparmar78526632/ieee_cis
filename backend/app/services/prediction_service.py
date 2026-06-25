import random

from app.services.model_loader import ModelLoader
from app.utils.logger import logger


class PredictionService:

    @staticmethod
    def predict(data):

        """
        Temporary Prediction Logic

        Later this will be replaced
        by Random Forest/LSTM.
        """

        # Check whether Random Forest model exists

        if ModelLoader.random_forest is not None:

            # Real prediction will come here later

            prediction = 0

        else:

            prediction = round(
                random.uniform(120, 980),
                2
            )

        confidence = round(
            random.uniform(90, 99),
            2
        )

        result = {

            "predicted_power": prediction,

            "confidence": confidence,

            "model": "Random Forest (Demo)",

            "status": "Prediction Successful"

        }

        logger.info(
            f"Prediction Generated | Power={prediction} | Confidence={confidence}"
        )

        return result