import pandas as pd
import os

DATASET_PATH = "app/dataset/Enhanced_Solar_Dataset.csv"


class AnalyticsService:

    @staticmethod
    def load_dataset():

        if not os.path.exists(DATASET_PATH):
            return None

        return pd.read_csv(DATASET_PATH)

    # -----------------------------------
    # Dataset Summary
    # -----------------------------------
    @staticmethod
    def get_summary():

        df = AnalyticsService.load_dataset()

        if df is None:
            return {
                "status": False,
                "message": "Dataset not found"
            }

        return {

            "rows": len(df),

            "columns": len(df.columns),

            "missing_values": int(df.isnull().sum().sum()),

            "duplicates": int(df.duplicated().sum()),

            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum()/1024/1024,
                2
            )

        }

    # -----------------------------------
    # Numerical Columns
    # -----------------------------------
    @staticmethod
    def get_numeric_columns():

        df = AnalyticsService.load_dataset()

        numeric = df.select_dtypes(include=["number"])

        return list(numeric.columns)

    # -----------------------------------
    # Correlation Matrix
    # -----------------------------------
    @staticmethod
    def correlation():

        df = AnalyticsService.load_dataset()

        numeric = df.select_dtypes(include=["number"])

        corr = numeric.corr()

        return corr.round(3).to_dict()

    # -----------------------------------
    # Feature Statistics
    # -----------------------------------
    @staticmethod
    def feature_statistics(column):

        df = AnalyticsService.load_dataset()

        if column not in df.columns:

            return {
                "status": False,
                "message": "Column not found"
            }

        return {

            "mean": float(df[column].mean()),

            "min": float(df[column].min()),

            "max": float(df[column].max()),

            "std": float(df[column].std()),

            "median": float(df[column].median())
        }