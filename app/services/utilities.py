import logging
from pathlib import Path

from typing import Optional

import librosa
import numpy as np
from pydub import AudioSegment
from pyrubberband import pyrb
import os
import shutil
import hashlib

from app.config.path_config import get_time_stretched_audio_path
from app.stores.song_cache import SongCache
from app.analysis.Segment import Segment

logger = logging.getLogger(__name__)


# supersedes adjust_tempo
def adjust_tempo(audio: np.ndarray, sr: int, playback_rate: float):
    logger.debug(f"Adjusting tempo with playback rate {playback_rate}")
    stretched_audio = pyrb.time_stretch(audio, sr, playback_rate)
    return stretched_audio


def adjust_tempo_and_recalculate_beats(session_id: str,
                                       audio: Optional[np.ndarray],
                                       sr: int,
                                       playback_rate: float,
                                       target_tempo: float,
                                       beats: np.ndarray,
                                       unique_name: Optional[str],
                                       in_memory_cache: SongCache,
                                       quantized_percussion: Optional[np.ndarray] = None,
                                       use_cache: bool = False) -> (
        Optional[np.ndarray], np.ndarray, float, Optional[float], Optional[np.ndarray]):
    logger.debug(
        f"Adjusting tempo of audio, beats and percussion, if given, with playback rate {playback_rate} of {unique_name}")
    assert playback_rate != 1.0

    stretched_audio = None
    stretched_audio_length = None

    time_stretched_audio_path = get_time_stretched_audio_path(session_id)
    stretched_audio_path = f'{time_stretched_audio_path}/{unique_name}_stretched_{playback_rate}.npy'
    exists_stretched_audio_in_cache = os.path.exists(stretched_audio_path)
    exists_stretched_audio_in_memory_cache = in_memory_cache.contains(stretched_audio_path)
    if audio is not None or (use_cache and exists_stretched_audio_in_cache):
        if use_cache:
            # check if the audio has already been stretched
            if exists_stretched_audio_in_memory_cache:
                logger.debug(f"Loading stretched audio {unique_name} from in-memory cache")
                stretched_audio = in_memory_cache.get_song(stretched_audio_path)
            elif exists_stretched_audio_in_cache:
                logger.debug(f"Loading stretched audio {unique_name} from cache")
                stretched_audio = np.load(stretched_audio_path)
                in_memory_cache.add_song(stretched_audio_path, stretched_audio)
            else:
                logger.debug(f"Stretching audio {unique_name} and caching it")
                stretched_audio = adjust_tempo(audio, sr, playback_rate)
                assert_time_stretched_audio_path_exists(time_stretched_audio_path)
                np.save(stretched_audio_path, stretched_audio)
                in_memory_cache.add_song(stretched_audio_path, stretched_audio)
        else:
            logger.debug(f"Stretching audio {unique_name}")
            stretched_audio = adjust_tempo(audio, sr, playback_rate)
        stretched_audio_length = get_audio_length(stretched_audio, sr)

    # apply time stretching to beats
    stretch_factor_beats = 1 / playback_rate
    stretched_beats = np.copy(beats)
    stretched_beats[:, 0] = stretched_beats[:, 0] * stretch_factor_beats

    stretched_quantized_percussion: Optional[np.ndarray] = None
    # apply time stretching to quantized percussion
    if quantized_percussion is not None:
        stretched_quantized_percussion = np.copy(quantized_percussion)
        stretched_quantized_percussion[:, 0] = stretched_quantized_percussion[:, 0] * stretch_factor_beats

    # calculate tempo
    stretched_tempo = target_tempo

    return stretched_audio, stretched_beats, stretched_tempo, stretched_audio_length, stretched_quantized_percussion


def time_stretch_segments(segments: list[Segment], playback_rate: float):
    stretch_factor_beats = 1 / playback_rate
    for segment in segments:
        segment.start_time = segment.start_time * stretch_factor_beats
        segment.end_time = segment.end_time * stretch_factor_beats


def time_stretch_interval_list_and_return(interval_list: list[tuple[int]],
                                          playback_factor: float) -> list[tuple[int, int]]:
    """
    Stretch the intervals in the list by the given playback factor.
    :param interval_list: List of intervals to stretch.
    :param playback_factor: Playback factor to use for stretching.
    :return: Stretched interval list.
    """
    stretch_factor = 1 / playback_factor
    return [(int(start * stretch_factor), int(end * stretch_factor)) for start, end in interval_list]


def truncate_to_n_decimals(value, n_decimals):
    return int(value * 10 ** n_decimals) / 10 ** n_decimals


def calculate_playback_rate(target_tempo: float, current_tempo: float) -> float:
    # adjust speed of candidate audio to match target tempo
    tempo_difference = abs(target_tempo - current_tempo)
    tempo_difference_double_target = abs(target_tempo - current_tempo / 2)
    tempo_difference_half_target = abs(target_tempo - current_tempo * 2)

    smallest_difference = min(tempo_difference, tempo_difference_half_target, tempo_difference_double_target)

    stretch_factor = 1.0
    if smallest_difference == tempo_difference:
        stretch_factor = target_tempo / current_tempo

    elif smallest_difference == tempo_difference_half_target:
        logger.debug(
            f'Difference between base ({current_tempo}) and candidate tempo ({target_tempo}) is half of target tempo')
        stretch_factor = target_tempo / (current_tempo * 2)
    elif smallest_difference == tempo_difference_double_target:
        logger.debug(
            f'Difference between base ({current_tempo}) and candidate tempo ({target_tempo}) is double of target tempo')
        stretch_factor = target_tempo / (current_tempo / 2)
    logger.debug(f'Calculated stretch factor: {stretch_factor}')

    return stretch_factor


def get_file_name_without_extension(path):
    base_name = os.path.basename(path)  # Get the base name
    file_name_without_extension = os.path.splitext(base_name)[0]  # Remove the extension
    return file_name_without_extension


def get_filename_with_hash_without_extension(file_path: str):
    file_name_without_extension = get_file_name_without_extension(file_path)
    file_hash = calculate_file_hash(file_path)
    return file_name_without_extension + '_' + file_hash


def get_filename_with_hash(file_path: str):
    filename_with_hash_without_extension = get_filename_with_hash_without_extension(file_path)
    file_extension = get_file_extension(file_path)
    return filename_with_hash_without_extension + file_extension


def get_file_extension(path):
    base_name = os.path.basename(path)  # Get the base name
    file_extension = os.path.splitext(base_name)[1]  # Remove the extension
    return file_extension


def copy_file(source: str, destination: str):
    shutil.copy2(source, destination)


def calculate_file_hash(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 16K
        for byte_block in iter(lambda: f.read(16384), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def get_audio_length(audio: np.ndarray, sr: int) -> float:
    if np.ndim(audio) == 1:
        return len(audio) / sr
    elif np.ndim(audio) == 2:
        return len(audio[0]) / sr
    else:
        raise RuntimeError('Audio should have either 1 or 2 dimensions')


def get_audio_length_from_file(audio_path: str) -> float:
    return librosa.get_duration(path=audio_path)


def get_beat_amount(audio: np.ndarray, sr: int, tempo: float) -> float:
    return get_audio_length(audio, sr) / (60 / tempo)


def get_duration_by_beats(beat_amount, tempo):
    return beat_amount * (60 / tempo)


def list_all_songs_in_folder(folder: str) -> list[str]:
    """
    Lists all the songs in a folder
    Supports .mp3, .wav, .flac and .m4a
    :param folder: The folder to list the songs from
    :return: A list of all the songs in the folder
    """

    songs = []
    for file in os.listdir(folder):
        if file.endswith(".mp3") or file.endswith(".wav") or file.endswith(".flac") or file.endswith(".m4a"):
            songs.append(os.path.join(folder, file))
    return songs


def convert_audio_array_of_samples_to_audio_segment(audio_vector: np.ndarray, sample_width: int, frame_rate: int,
                                                    channels: int) -> AudioSegment:
    if sample_width == 1:  # 8-bit audio
        audio_vector = np.int8(audio_vector)
    elif sample_width == 2:  # 16-bit audio
        audio_vector = np.int16(audio_vector)
    elif sample_width == 3:  # 24-bit audio (note that this is less common)
        audio_vector = np.int32(audio_vector)
    elif sample_width == 4:  # 32-bit audio
        audio_vector = np.int32(audio_vector)
    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    return AudioSegment(audio_vector.tobytes(), frame_rate=frame_rate, sample_width=sample_width, channels=channels)


def time_stretch_rubberband(y: np.ndarray, sr: int, playback_rate: float):
    return pyrb.time_stretch(y, sr, playback_rate)


def pitch_shift_rubberband(y: np.ndarray, sr: int, pitch_shift_semitones: int):
    return pyrb.pitch_shift(y, sr, pitch_shift_semitones)


def adjust_tempo_and_key_audio_segment(audio: AudioSegment, playback_rate: float, key_shift: int) -> AudioSegment:
    logger.debug(
        f"Adjusting tempo and key of audio segment with playback rate {playback_rate} and key shift {key_shift}")

    if audio.sample_width == 1:
        int_type = np.int8
    elif audio.sample_width == 2:
        int_type = np.int16
    elif audio.sample_width == 3 or audio.sample_width == 4:
        int_type = np.int32
    else:
        raise ValueError(f"Unsupported sample_width: {audio.sample_width}")

    channels = []
    for chn in range(audio.channels):
        # extract one channel of samples
        samples = np.array(audio.get_array_of_samples()[chn::audio.channels])
        # convert the samples to float in range [-1, 1]
        samples = samples.astype(np.float32)
        samples /= np.iinfo(int_type).max
        channels.append(samples)

    # sox sounds better but fails for larger semitone shifts, rubberband significantly outperforms sox for larger
    # --> UPDATE: sox introduces weird artifacts, so we continue to use rubberband for all pitch shifts
    # pitch_shift = pitch_shift_sox if abs(key_shift) <= 2 else pitch_shift_rubberband
    pitch_shift = pitch_shift_rubberband

    audio_np = np.column_stack(channels)
    audio_np_stretched = time_stretch_rubberband(audio_np, audio.frame_rate,
                                                 playback_rate) if playback_rate != 1.0 else audio_np
    audio_np_stretched_key_shifted = pitch_shift(audio_np_stretched, audio.frame_rate,
                                                 key_shift) if key_shift != 0 else audio_np_stretched
    # interleave channels
    audio_np_stretched_key_shifted_flatten = audio_np_stretched_key_shifted.flatten()
    # convert back to integer samples
    audio_np_stretched_key_shifted_flatten *= np.iinfo(int_type).max
    audio_np_stretched_key_shifted_flatten = np.round(audio_np_stretched_key_shifted_flatten).astype(int_type)

    # Create a new AudioSegment from the stretched data
    final_audio = AudioSegment(
        data=audio_np_stretched_key_shifted_flatten.tobytes(),
        sample_width=audio.sample_width,
        frame_rate=audio.frame_rate,
        channels=audio.channels
    )
    return final_audio


def assert_time_stretched_audio_path_exists(time_stretched_audio_path: str):
    analysis_path = Path(os.path.join(time_stretched_audio_path))
    analysis_path.mkdir(parents=True, exist_ok=True)
