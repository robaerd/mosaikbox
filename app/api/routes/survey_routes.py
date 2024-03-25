from fastapi import APIRouter, status

from app.models.survey_response import SurveyResponse
import logging

from app.services.survey_service import SurveyService

router = APIRouter()
survey_service = SurveyService()
logger = logging.getLogger(__name__)


@router.post("/survey", status_code=status.HTTP_201_CREATED)
async def finish_survey(survey_response: SurveyResponse):
    logger.info(f"Finish survey")
    logger.debug(f"Survey response: {survey_response}")

    survey_service.finish_survey(survey_response)
