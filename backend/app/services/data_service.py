import pandas as pd
import os
from app.utils.logger import logger
DATASET_PATH = "app/dataset/Enhanced_Solar_Dataset.csv"


class DatasetService:

    @staticmethod
    def load_dataset():

        if not os.path.exists(DATASET_PATH):
            return None

        return pd.read_csv(DATASET_PATH)
        logger.info("Dataset Loaded Successfully")

    @staticmethod
    def get_dataset_info():

        df = DatasetService.load_dataset()

        if df is None:
            return {
                "status": False,
                "message": "Dataset not found."
            }

        return {

            "status": True,

            "rows": int(df.shape[0]),

            "columns": int(df.shape[1]),

            "missing_values": int(df.isnull().sum().sum()),

            "duplicates": int(df.duplicated().sum()),

            "column_names": list(df.columns)
        }

    @staticmethod
    def get_statistics():

        df = DatasetService.load_dataset()

        if df is None:
            return {
                "status": False,
                "message": "Dataset not found."
            }

        return df.describe().to_dict()