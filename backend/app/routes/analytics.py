from fastapi import APIRouter

from app.services.analytics_service import AnalyticsService

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


@router.get("/summary")
def summary():

    return AnalyticsService.get_summary()


@router.get("/columns")
def columns():

    return AnalyticsService.get_numeric_columns()


@router.get("/correlation")
def correlation():

    return AnalyticsService.correlation()


@router.get("/statistics/{column}")
def statistics(column: str):

    return AnalyticsService.feature_statistics(column)