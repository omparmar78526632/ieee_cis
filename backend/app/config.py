import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()


class Settings:

    APP_NAME = os.getenv(
        "APP_NAME",
        "Solar Generation Prediction API"
    )

    APP_VERSION = os.getenv(
        "APP_VERSION",
        "1.0.0"
    )

    HOST = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    PORT = int(
        os.getenv(
            "PORT",
            8000
        )
    )

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///solar.db"
    )


settings = Settings()