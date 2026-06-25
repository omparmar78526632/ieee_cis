from fastapi import APIRouter

from app.services.data_service import DatasetService

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"]
)


@router.get("/info")
def dataset_info():

    return DatasetService.get_dataset_info()


@router.get("/statistics")
def dataset_statistics():

    return DatasetService.get_statistics()