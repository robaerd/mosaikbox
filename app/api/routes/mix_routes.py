from fastapi import APIRouter, status, UploadFile, File, BackgroundTasks
from starlette.responses import FileResponse

from app.models.mix import MixInitiateResponse, AnalysisStartRequest, MixStartRequest
from app.services.mix_service import MixService
import logging

router = APIRouter()
mix_service = MixService()
logger = logging.getLogger(__name__)


@router.post("/mix/", response_model=MixInitiateResponse, status_code=status.HTTP_201_CREATED)
async def request_session():
    logger.info("Creating mix session")
    return mix_service.start_session()


@router.post("/mix/{session_id}", status_code=status.HTTP_201_CREATED)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    logger.info(f"Uploading file {file.filename} to session {session_id}")
    return mix_service.upload_file(session_id, file)


@router.put("/mix/{session_id}", status_code=status.HTTP_202_ACCEPTED)
async def start_mix(session_id: str, analysis_start_request: AnalysisStartRequest, background_tasks: BackgroundTasks):
    logger.info(f"Starting mix for session {session_id}")
    background_tasks.add_task(mix_service.start_analysis, session_id, analysis_start_request)
    return {"info": f"Mix started for session {session_id}"}


@router.put("/mix/{session_id}/start_mix", status_code=status.HTTP_202_ACCEPTED)
async def start_mix(session_id: str, mix_start_request: MixStartRequest, background_tasks: BackgroundTasks):
    logger.info(f"Starting mix for session {session_id}")
    background_tasks.add_task(mix_service.start_mixing, session_id, mix_start_request)
    return {"info": f"Mix started for session {session_id}"}


@router.get("/mix/{session_id}/status", status_code=status.HTTP_200_OK)
async def get_mix_status(session_id: str):
    logger.info(f"Getting mix status for session {session_id}")
    return mix_service.get_status(session_id)


@router.get("/mix/{session_id}/schedule", status_code=status.HTTP_200_OK)
async def get_mix_schedule(session_id: str):
    logger.info(f"Getting mix schedule for session {session_id}")
    return mix_service.get_song_schedule(session_id)


@router.get("/mix/{session_id}", status_code=status.HTTP_200_OK)
async def get_mix(session_id: str):
    logger.info(f"Getting mix for session {session_id}")
    mix_filepath = mix_service.get_mix_filepath(session_id)
    return FileResponse(mix_filepath, filename=mix_filepath)


@router.delete("/mix/{session_id}", status_code=status.HTTP_200_OK)
async def end_mix(session_id: str):
    logger.info(f"Ending mix session {session_id}")
    return mix_service.end_session(session_id)