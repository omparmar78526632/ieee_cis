from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy import DateTime

from datetime import datetime

from app.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    temperature = Column(Float)

    humidity = Column(Float)

    pressure = Column(Float)

    wind_speed = Column(Float)

    irradiation = Column(Float)

    predicted_power = Column(Float)

    confidence = Column(Float)

    model_name = Column(String)