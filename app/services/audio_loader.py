import logging

import numpy as np
from librosa import load as librosa_load
from pyloudnorm import Meter
from tinytag import TinyTag

from .utilities import get_audio_length

logger = logging.getLogger(__name__)


def load(path: str, sr: int = 44100, mono: bool = True, eql_loudness: bool = True) -> (np.ndarray, int, float):
    """
    Load an audio file from disk and return the audio data, sample rate and length in seconds
    :param path: path to the audio file
    :param sr: sample rate
    :param mono: whether to load the audio in mono
    :param eql_loudness: whether to normalize the loudness to a target LUFS

    :return: audio data, sample rate, audio length in seconds
    """
    audio, sr = librosa_load(path, sr=sr, mono=mono)
    audio_length = get_audio_length(audio, int(sr))

    # normalize audio
    if eql_loudness:
        audio, _ = normalize_loudness(audio, int(sr))

    return audio, sr, audio_length


def normalize_loudness(audio: np.ndarray, sr: int, target_loudness_lufs=-14) -> tuple[np.ndarray, float]:
    """
    Normalize the loudness of the audio to a target loudness level
    We use -14 LUFS as the target loudness level, as it is the standard for streaming services

    :param audio: audio data
    :param sr: sample rate
    :param target_loudness_lufs: target loudness level in LUFS

    :return normalized audio data, gain applied to the audio data
    """
    logger.debug(f"Normalizing loudness to {target_loudness_lufs} LUFS")

    meter = Meter(sr)
    loudness = meter.integrated_loudness(audio)
    logger.debug(f"Detected loudness: {loudness} LUFS")

    # calculate the required gain to achieve the target loudness
    gain = target_loudness_lufs - loudness
    logger.debug(f"Required gain: {gain}")

    # apply the gain to the audio data
    normalized_data = apply_gain_lufs(audio, gain)
    return normalized_data, gain


def apply_gain_lufs(audio: np.ndarray, gain: float) -> np.ndarray:
    return audio * 10 ** (gain / 20)


def get_title_artist_of_audio_file(path: str) -> (str, str):
    # read the tags from the file
    tag = TinyTag.get(path)

    title = tag.title
    artist = tag.artist if tag.artist is not None else tag.albumartist

    return title, artist
