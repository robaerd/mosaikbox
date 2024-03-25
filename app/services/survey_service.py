import logging

from mongoengine import connect

from app.config import config
from app.documents.survey_document import SurveyDocument
from app.models.survey_response import SurveyResponse

logger = logging.getLogger(__name__)


class SurveyService:

    def __init__(self):
        connect(
            db=config.MONGODB_DB,
            username=config.MONGODB_USER,
            password=config.MONGODB_PASSWORD,
            host=config.MONGODB_URL,
            port=config.MONGODB_PORT
        )

    def finish_survey(self, survey_response: SurveyResponse):
        logger.debug("saving survey")
        # Convert the Pydantic model to a dictionary
        survey_response_dict = survey_response.dict()
        # Create a new SurveyResponse document
        survey_response_doc = SurveyDocument(**survey_response_dict)
        # Save the document into the MongoDB collection
        survey_response_doc.save()
