import logging
import os

from enum import Enum
from pathlib import Path

import demucs.separate
from pydub import AudioSegment

from .utilities import get_file_name_without_extension, get_filename_with_hash_without_extension
from app.config.config import DEMUCS_MODEL
from app.config.path_config import get_stem_path, get_stem_mixed_path
from .audio_loader import load

logger = logging.getLogger(__name__)


class StemType(Enum):
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    VOCALS = "vocals"


def separate(session_id: str, song_path: str):
    logger.debug(f"Separating stems for song {song_path}")
    stem_path = get_stem_path(session_id)
    assert_stem_path_exists(stem_path)
    song_name_with_hash = get_filename_with_hash_without_extension(song_path)

    filename = song_name_with_hash + "/{stem}.{ext}"
    # todo: add device detection and remove hardcoded "mps" device so this also runs under linux and other platforms
    demucs.separate.main(["-n", DEMUCS_MODEL, "-d", "mps", "-o", stem_path, "--filename", filename, song_path])


def get_stem_for_song(session_id, song_path: str, stem_type: StemType) -> AudioSegment:  # (np.ndarray, int, float):
    """
    :param song_path: path to the song
    :param stem_type: the type of stem to get
    :return: the stem as a numpy array and the sample rate
    """
    logger.debug(f"Loading stem {stem_type} for song {song_path}")
    stem_path = get_song_stem_filepath(session_id, song_path, stem_type)
    return AudioSegment.from_file(stem_path)


def mix_song_without_stem(session_id, song_path, stem_type_to_exclude: list[StemType]):
    logger.debug(f"Mixing song {song_path} without {stem_type_to_exclude} stem")
    stem_mixed_path = get_stem_mixed_path(session_id)
    assert_stem_mixed_math_exists(stem_mixed_path)

    stem_type_to_exclude = sorted(stem_type_to_exclude, key=lambda stem_type: stem_type.value)

    all_stem_types = [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]
    song_stem_types = [stem_type for stem_type in all_stem_types if stem_type not in stem_type_to_exclude]

    stem_type_to_exclude_name_concat = '_'.join([stem_type.value for stem_type in stem_type_to_exclude])
    stem_excluded_song_path = os.path.join(stem_mixed_path, get_file_name_without_extension(
        song_path) + "_without_" + stem_type_to_exclude_name_concat + ".wav")
    if os.path.exists(stem_excluded_song_path):
        logger.debug(f"Song {song_path} without {stem_type_to_exclude} stem already exists, skipping...")
        return

    audio = None
    for stem_type in song_stem_types:
        stem = get_stem_for_song(session_id, song_path, stem_type)
        if audio is None:
            audio = stem
        else:
            audio = audio.overlay(stem)
    audio.export(stem_excluded_song_path, format="wav")


def get_song_without_stem_filepath(session_id: str, song_path, stem_type_to_exclude: list[StemType]):
    logger.debug(f"Getting filepath of mixed song {song_path} without {stem_type_to_exclude} stem")
    stem_mixed_path = get_stem_mixed_path(session_id)
    stem_type_to_exclude = sorted(stem_type_to_exclude, key=lambda stem_type: stem_type.value)
    stem_type_to_exclude_name_concat = '_'.join([stem_type.value for stem_type in stem_type_to_exclude])
    return os.path.join(stem_mixed_path,
                        get_file_name_without_extension(
                            song_path) + "_without_" + stem_type_to_exclude_name_concat + ".wav")


def get_song_stem_filepath(session_id: str, song_path, stem_type: StemType) -> str:
    logger.debug(f"Getting filepath of {stem_type} stem for song {song_path}")
    stem_path = get_stem_path(session_id)
    audio_name_without_extension = get_filename_with_hash_without_extension(song_path)
    stem_path = os.path.join(stem_path, DEMUCS_MODEL, audio_name_without_extension, stem_type.value + ".wav")
    return stem_path


def get_song_without_stem(session_id, song_path, stem_type_to_exclude: StemType, sr: int = 44100):
    logger.debug(f"Loading song {song_path} without {stem_type_to_exclude} stem")
    song_without_stem_path = get_song_without_stem_filepath(session_id, song_path, stem_type_to_exclude)
    if os.path.exists(song_without_stem_path):
        logger.debug(f"Song {song_path} without {stem_type_to_exclude} stem already exists, loading...")
        return load(song_without_stem_path, sr=sr)
    else:
        logger.debug(f"Song {song_path} without {stem_type_to_exclude} stem does not exist, creating...")
        mix_song_without_stem(session_id, song_path, stem_type_to_exclude)
        return load(song_without_stem_path, sr=sr)


def assert_stem_path_exists(stem_path: str):
    analysis_path = Path(os.path.join(stem_path))
    analysis_path.mkdir(parents=True, exist_ok=True)


def assert_stem_mixed_math_exists(stem_mixed_path: str):
    analysis_path = Path(os.path.join(stem_mixed_path))
    analysis_path.mkdir(parents=True, exist_ok=True)
