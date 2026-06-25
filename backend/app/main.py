from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.prediction import router as prediction_router
from app.database import engine
from app import models
from app.routes.dataset import router as dataset_router
from app.routes.analytics import router as analytics_router
from app.utils.logger import logger
from app.routes.decision import router as decision_router
from app.services.model_loader import ModelLoader
from app.utils.exception_handler import global_exception_handler

models.Base.metadata.create_all(bind=engine)
ModelLoader.load_models()

logger.info("Solar Generation Prediction API Started")

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for AI-Based Solar Generation Prediction and Decision Support System",
    version=settings.APP_VERSION
)
app.add_exception_handler(
    Exception,
    global_exception_handler
)
app.include_router(dataset_router)
app.include_router(prediction_router)
app.include_router(analytics_router)
app.include_router(decision_router)
# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Welcome to Solar Generation Prediction API",
        "status": "Running Successfully",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "server": "FastAPI",
        "project": "Solar Generation Prediction"
    }