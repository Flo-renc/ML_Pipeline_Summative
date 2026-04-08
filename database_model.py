from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base, engine

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String(255), nullable=False)
    predicted_character = Column(String(50), nullable=False)
    true_label = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=False)
    inference_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)



