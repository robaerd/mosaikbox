import logging
import os
from pathlib import Path

from app.config.config import COMPUTATION_PATH_PROD, COMPUTATION_PATH_DEV

logger = logging.getLogger(__name__)

ENV = os.environ.get("ENV")
if ENV == 'prod':
    COMPUTATION_PATH = COMPUTATION_PATH_PROD
else:
    COMPUTATION_PATH = COMPUTATION_PATH_DEV
# computation path has to be absolute
COMPUTATION_PATH = os.path.abspath(COMPUTATION_PATH)

logger.info(f"Using computation path: {COMPUTATION_PATH}")


def get_segmentation_data_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "segmentation")


def get_analysis_data_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "analysis")


def get_song_schedule_data_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "song_schedule")


def get_time_stretched_audio_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "time_stretched_audio")


def get_beat_sync_chroma_and_spectrum_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "beat_sync_chroma_and_spectrum")


def get_stem_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "stems")


def get_stem_mixed_path(session_id: str):
    return os.path.join(get_stem_path(session_id), "stems_mixed")


def get_lyrics_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "lyrics")


def get_uploaded_songs_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "uploaded_songs")


def get_generated_mix_path(session_id: str):
    return os.path.join(COMPUTATION_PATH, session_id, "generated_mix")


def _convert_to_absolute(path_string):
    path = Path(path_string)
    if not path.is_absolute():
        absolute_path = path.resolve()
        return absolute_path
    else:
        return path
