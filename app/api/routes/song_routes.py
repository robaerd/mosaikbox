from fastapi import APIRouter, status, UploadFile, File, BackgroundTasks
from starlette.responses import FileResponse

import logging

from app.services.song_service import SongService

router = APIRouter()
song_service = SongService()
logger = logging.getLogger(__name__)


@router.get("/song/{session_id}/{song_name}", status_code=status.HTTP_200_OK)
async def get_mix(session_id, song_name: str):
    logger.info(f"Getting song {song_name} for session {session_id}")
    song_filepath = song_service.get_song_filepath(session_id, song_name)
    return FileResponse(song_filepath, filename=song_filepath)
