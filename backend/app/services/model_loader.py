import os
import joblib

from tensorflow.keras.models import load_model


class ModelLoader:

    random_forest = None

    lstm_model = None

    scaler = None


    @classmethod
    def load_models(cls):

        # Random Forest

        rf_path = "app/trained_models/random_forest.pkl"

        if os.path.exists(rf_path):

            cls.random_forest = joblib.load(rf_path)

            print("✅ Random Forest Loaded")

        else:

            print("⚠ Random Forest model not found")


        # LSTM

        lstm_path = "app/trained_models/lstm.keras"

        if os.path.exists(lstm_path):

            cls.lstm_model = load_model(lstm_path)

            print("✅ LSTM Loaded")

        else:

            print("⚠ LSTM model not found")


        # Scaler

        scaler_path = "app/trained_models/scaler.pkl"

        if os.path.exists(scaler_path):

            cls.scaler = joblib.load(scaler_path)

            print("✅ Scaler Loaded")

        else:

            print("⚠ Scaler not found")