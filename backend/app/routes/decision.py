from fastapi import APIRouter

from app.schemas import DecisionInput

from app.services.decision_service import DecisionService

router = APIRouter(

    prefix="/decision",

    tags=["Decision Support"]

)


@router.post("/recommend")
def recommend(data: DecisionInput):

    return DecisionService.generate_recommendation(data)