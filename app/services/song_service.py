import logging
import os

from app.config import path_config
from app.exceptions.custom_exceptions import NotFoundException

logger = logging.getLogger(__name__)


class SongService:
    def get_song_filepath(self, session_id: str, song_name: str) -> str:
        logger.debug(f"Getting mix filepath for session {session_id}")
        song_filepath = os.path.join(path_config.get_uploaded_songs_path(session_id), song_name)
        if not os.path.exists(song_filepath):
            raise NotFoundException(object_type=f"Song {song_name}", object_id=session_id)
        return song_filepath
