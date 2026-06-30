import numpy as np

from app.services.model_loader import ModelLoader
from app.utils.logger import logger


class PredictionService:

    @staticmethod
    def predict(data):

        """
        Predict AC power output from weather inputs.

        Features (order must match training):
            [AMBIENT_TEMPERATURE, RH2M, PS, WS10M, IRRADIATION]
            i.e. [temperature, humidity, pressure, wind_speed, irradiation]
        """

        if ModelLoader.random_forest is not None and ModelLoader.scaler is not None:

            # ── Real model inference ──────────────────────────────
            features = np.array([[
                data.temperature,
                data.humidity,
                data.pressure,
                data.wind_speed,
                data.irradiation,
            ]])

            features_scaled = ModelLoader.scaler.transform(features)

            raw_prediction = float(
                ModelLoader.random_forest.predict(features_scaled)[0]
            )

            # Clamp to non-negative (solar power cannot be negative)
            prediction = round(max(0.0, raw_prediction), 2)

            model_label = "Trained ML Model"

            # Confidence: not directly available from sklearn regressors,
            # so we approximate via out-of-bag score if RF, else fixed 95
            if hasattr(ModelLoader.random_forest, "oob_score_"):
                confidence = round(ModelLoader.random_forest.oob_score_ * 100, 2)
            else:
                confidence = 95.0

        else:

            # ── Fallback: model not loaded ────────────────────────
            logger.warning("ML model not loaded — using irradiation-based estimate")

            # Physics-based fallback: AC_POWER estimation ≈ irradiation * area * efficiency
            # Using typical 1kW panel params as rough estimate
            prediction = round(max(0.0, data.irradiation * 8500), 2)
            confidence = 60.0
            model_label = "Physics Estimate (Model Not Loaded)"

        result = {
            "predicted_power": prediction,
            "confidence": confidence,
            "model": model_label,
            "status": "Prediction Successful",
        }

        logger.info(
            f"Prediction | Power={prediction}W | Confidence={confidence}% | Model={model_label}"
        )

        return result