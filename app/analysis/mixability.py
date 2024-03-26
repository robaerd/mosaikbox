import logging
from pathlib import Path
from torch import Tensor

import librosa.core as lr_core
import librosa.feature as lr_feature
import numpy as np
from scipy import signal
from app.key_estimator.key_distance import key_distance_harmonic
from app.key_estimator.key_similarity import calculate_key_similarity
import hashlib
import pickle
import os

from . import timbre_calculation
from .timbre_similarity import calculate_timbre_similarity
from .text_similarity import TextSimilarity

import app.analysis.rhythmic_similarity as rs
from app.config.path_config import get_beat_sync_chroma_and_spectrum_path

logger = logging.getLogger(__name__)


class ShorterException(Exception):
    pass


def calc_mixability(session_id: str,
                    audio1_vector: np.ndarray,
                    audio2_vector: np.ndarray,
                    audio1_percussion: np.ndarray,
                    audio2_percussion: np.ndarray,
                    audio1_beats: np.ndarray,
                    audio2_beats: np.ndarray,
                    audio1_key: str,
                    audio2_key: str,
                    audio2_beat_sync_timbre: list[np.ndarray],
                    audio1_sr=44100,
                    audio2_sr=44100,
                    enable_contextual_similarity: bool = False,
                    audio1_lyrics_embedding: Tensor = None,
                    audio2_lyrics_embedding: Tensor = None):
    assert audio1_percussion[0, 0] == 0.0
    assert audio1_beats[0, 0] == 0.0

    assert audio2_percussion[0, 0] == 0.0
    assert audio2_beats[0, 0] == 0.0

    beat_sync_chroma_and_spectrum_path = get_beat_sync_chroma_and_spectrum_path(session_id)
    base_beat_sync_chroma, base_beat_sync_spectrum = _calc_beat_sync_chroma_and_spectrum(audio1_vector, sr=audio1_sr,
                                                                                         beats=audio1_beats,
                                                                                         beat_sync_chroma_and_spectrum_path=beat_sync_chroma_and_spectrum_path)

    cand_beat_sync_chroma, cand_beat_sync_spectrum = _calc_beat_sync_chroma_and_spectrum(audio2_vector, sr=audio2_sr,
                                                                                         beats=audio2_beats,
                                                                                         beat_sync_chroma_and_spectrum_path=beat_sync_chroma_and_spectrum_path)

    base_beat_length = base_beat_sync_spectrum.shape[1]
    n_max_b_shifts = cand_beat_sync_spectrum.shape[1] - base_beat_sync_spectrum.shape[1]
    if n_max_b_shifts < 0:
        raise ShorterException("Candidate song has lesser beats than base song")

    # Calculate M_H
    harmonic_sim_matrix, harmonic_sim_k = _calc_harmonic_similarity(base_beat_sync_chroma, cand_beat_sync_chroma)
    # Calculate M_R
    rhythm_sim_k = _calc_rhythmic_similarity(audio1_percussion, audio2_percussion, audio2_beats, n_max_b_shifts,
                                             base_beat_length)
    # Calculate M_L
    spectral_balance_k = _calc_spectral_balance(base_beat_sync_spectrum, cand_beat_sync_spectrum, base_beat_length,
                                                n_max_b_shifts)

    # Calculate mixability
    mixability = 0.2 * rhythm_sim_k + 1.0 * harmonic_sim_k + spectral_balance_k * 0.2
    b_offset = np.argmax(mixability)

    # AMU pitch shift approach
    p_shift = np.argmax(harmonic_sim_matrix[:, b_offset])
    p_shift = p_shift - 6
    p_shift = -p_shift

    return np.max(
        mixability), p_shift, p_shift, b_offset, harmonic_sim_k, spectral_balance_k, rhythm_sim_k, None, None, mixability


def _calc_harmonic_similarity(base_beat_sync_chroma: np.ndarray, candidate_beat_sync_chroma: np.ndarray) -> (
        np.ndarray, np.ndarray):
    # flip for cross-correlation
    cand_beat_sync_chroma_flipped = np.flip(candidate_beat_sync_chroma)
    cand_beat_sync_chroma_flipped_stack = np.concatenate((cand_beat_sync_chroma_flipped, cand_beat_sync_chroma_flipped),
                                                         axis=0)
    # Perform 2D cross-correlation
    convolution_result = signal.convolve2d(cand_beat_sync_chroma_flipped_stack, base_beat_sync_chroma, mode='full')
    norm_base, norm_candidate = np.linalg.norm(base_beat_sync_chroma), np.linalg.norm(candidate_beat_sync_chroma)
    normalized_convolution = convolution_result / (norm_base * norm_candidate)
    # Calculate the adjustment for slicing based on the base chroma width
    slice_adjustment = base_beat_sync_chroma.shape[1] - 1
    harmonic_sim_matrix = np.flipud(normalized_convolution[11:-11, slice_adjustment:-slice_adjustment])
    harmonic_sim_k = np.amax(harmonic_sim_matrix, axis=0)
    return harmonic_sim_matrix, harmonic_sim_k


def _calc_rhythmic_similarity(audio1_percussion, audio2_percussion, audio2_beats, n_max_b_shifts, base_beat_length):
    rhythm_sim_k = np.zeros(n_max_b_shifts + 1)

    for i in range(n_max_b_shifts):
        # we extract the percusion based on the beats as percussion are quantized over 12 equal positions over each beat
        beat_start_timing = audio2_beats[i, 0]
        beat_end_timing = audio2_beats[i + base_beat_length, 0]
        audio2_percussion_k = rs.get_subset_of_percussion_timing(audio2_percussion, beat_start_timing,
                                                                 beat_end_timing)
        # we exclude kick drum from percussive similarity
        rhythm_sim = rs.calculate_percussive_similarity(quantized_beats_1=audio1_percussion,
                                                        quantized_beats_2=audio2_percussion_k,
                                                        features=['snare', 'hihat', 'kick'])
        rhythm_sim_k[i] = rhythm_sim
    return rhythm_sim_k


def _calc_spectral_balance(base_beat_sync_spectrum, cand_beat_sync_spectrum, base_beat_length, n_max_b_shifts):
    spectral_balance_k = np.zeros(n_max_b_shifts + 1)
    for i in range(n_max_b_shifts + 1):
        beta = np.mean(base_beat_sync_spectrum + cand_beat_sync_spectrum[:, i:i + base_beat_length], axis=1)
        beta_norm = beta / np.sum(beta)
        spectral_balance_k[i] = 1 - np.std(beta_norm)
    return spectral_balance_k

def _calc_beat_sync_chroma_and_spectrum(eql_y, sr, beats, beat_sync_chroma_and_spectrum_path):
    try:
        # Attempt to load cached results
        beat_sync_chroma_and_spectrum = load_beat_sync_chroma_and_spectrum(
            eql_y, beats, sr, beat_sync_chroma_and_spectrum_path)
        logger.debug("Cache hit: Loaded beat_sync_chroma_and_spectrum")
        return beat_sync_chroma_and_spectrum
    except FileNotFoundError:
        logger.debug("Cache miss: Computing beat_sync_chroma_and_spectrum")

    # Initialize lists to collect band energies and chromas
    beat_pos = beats[:, 0]
    band_energies = [np.zeros(len(beat_pos) - 1) for _ in range(3)]
    chromas = []
    bands = [(0, 220), (220, 1760), (1760, sr / 2)]

    # Process each beat interval
    for i in range(1, len(beat_pos)):
        start_frame, end_frame = int(beat_pos[i - 1] * sr), int(beat_pos[i] * sr)
        beat_signal = eql_y[start_frame:end_frame]

        # Compute FFT once per beat interval
        fft_eq = np.abs(np.fft.fft(beat_signal))
        freqs = np.fft.fftfreq(len(fft_eq), 1 / sr)

        # Calculate energy in specified bands
        for j, (low_freq, high_freq) in enumerate(bands):
            mask = (freqs > low_freq) & (freqs < high_freq)
            band_energies[j][i - 1] = np.sqrt(np.mean(np.sum(fft_eq[mask] ** 2)))

        # Compute and accumulate chroma features
        stft = np.abs(lr_core.stft(beat_signal))
        chroma = np.mean(lr_feature.chroma_stft(y=None, S=stft ** 2), axis=1)
        chromas.append(chroma)

    # Combine results
    chromas = np.array(chromas).T
    beat_sync_chroma_and_spectrum = (chromas, np.vstack(band_energies))

    # Cache the computed results
    save_beat_sync_chroma_and_spectrum(eql_y, beats, sr, beat_sync_chroma_and_spectrum,
                                       beat_sync_chroma_and_spectrum_path)
    return beat_sync_chroma_and_spectrum


def calculate_audio_beats_sr_hash(audio: np.ndarray, beats: np.ndarray, sr: int):
    audio_bytes = audio.tobytes()
    beats_bytes = beats.tobytes()
    sr_bytes = sr.to_bytes((sr.bit_length() + 7) // 8, 'big')

    m = hashlib.sha256()
    m.update(audio_bytes)
    m.update(beats_bytes)
    m.update(sr_bytes)

    return m.hexdigest()


def save_beat_sync_chroma_and_spectrum(audio: np.ndarray, beats: np.ndarray, sr: int,
                                       beat_sync_chroma_and_spectrum: tuple[np.ndarray, np.ndarray],
                                       beat_sync_chroma_and_spectrum_path: str):
    hash_value = calculate_audio_beats_sr_hash(audio, beats, sr)
    # create path if not exists
    assert_beat_sync_chroma_and_spectrum_path_exists(beat_sync_chroma_and_spectrum_path)
    # pickle the result
    path = os.path.join(beat_sync_chroma_and_spectrum_path, hash_value + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(beat_sync_chroma_and_spectrum, f)


def load_beat_sync_chroma_and_spectrum(audio: np.ndarray, beats: np.ndarray, sr: int,
                                       beat_sync_chroma_and_spectrum_path: str) -> tuple[np.ndarray, np.ndarray]:
    hash_value = calculate_audio_beats_sr_hash(audio, beats, sr)
    path = os.path.join(beat_sync_chroma_and_spectrum_path, hash_value + '.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def assert_beat_sync_chroma_and_spectrum_path_exists(beat_sync_chroma_and_spectrum_path: str):
    analysis_path = Path(os.path.join(beat_sync_chroma_and_spectrum_path))
    analysis_path.mkdir(parents=True, exist_ok=True)
