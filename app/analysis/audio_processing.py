import logging
from typing import Optional
from torch import Tensor

import app.key_estimator.key_estimator_client as key_estimator
import app.services.audio_loader as audio_loader
import app.analysis.beats as bts
import app.analysis.structural_segmentation as ss
import app.analysis.rhythmic_similarity as rs
import app.analysis.AnalysisResult as AnalysisResult_ds
import app.services.utilities as utilities
import app.stores.analysis_store as analysis_store
import app.analysis.vocal as vocal
from app.analysis.text_similarity import TextSimilarity
from app.models.mix import MixModel
from app.services.lyrics_loader import LyricsLoader

logger = logging.getLogger(__name__)


def process_audio(audio_path, session_id: str, mix_model: MixModel, plot_segments=False):
    logger.info(f'Analyzing song {audio_path} for session {session_id}')
    if mix_model == MixModel.MOSAIKBOX:
        process_audio_default(audio_path, session_id, plot_segments)
    elif mix_model == MixModel.AMU:
        process_audio_amu(audio_path, session_id, plot_segments)


def process_audio_default(audio_path, session_id: str, plot_segments=False):
    """
    Mosaikbox is implemented in the main branch. You are on the AMU branch.
    """
    raise NotImplementedError("Mosaikbox is implemented in the main branch. You are on the AMU branch.")


def process_audio_amu(audio_path, session_id: str, plot_segments=False):
    logger.debug(f'Using amu analysis for song {audio_path} for session {session_id}')
    logger.debug(f'Using default analysis for song {audio_path} for session {session_id}')
    # load audio
    audio, sr, audio_length = audio_loader.load(audio_path, sr=44100)
    # estimate beat grid and tempo
    beats, tempo = bts.determine_beats_and_tempo(audio_path=audio_path, audio_length=audio_length)
    # calculate segments
    boundaries, labels = ss.calculate_song_boundaries(session_id=session_id, audio=audio, sr=sr,
                                                      audio_path=audio_path, beats=beats,
                                                      plot=plot_segments)
    segments = ss.process_song_boundaries_to_segment_timing(boundaries, labels, tempo, beats)
    segments = ss.calculate_segment_compatibility_based_on_labels(segments, audio_length=audio_length)
    downbeat_segments = ss.filter_and_move_segment_boundaries_to_downbeats(segments=segments, beats=beats, tempo=tempo)

    # transcribe percussion
    audio_percussion_quantized = rs.transcribe_percussion(audio, sr, beats[:, 0])

    # store analysis results to file
    analysis = AnalysisResult_ds.AnalysisResult(audio_path,
                                                filename_with_hash=utilities.get_filename_with_hash(audio_path),
                                                bpm=tempo,
                                                target_bpm=None,
                                                key=None,
                                                target_key=None,
                                                beats=beats,
                                                quantized_percussion=audio_percussion_quantized,
                                                downbeat_segments=downbeat_segments,
                                                sample_rate=sr,
                                                audio_length=audio_length,
                                                time_until_first_beat_stretched=None,
                                                stretch_playback_rate=None,
                                                lyrics_embedding=None,
                                                vocal_segments=[],
                                                mix_model=MixModel.AMU)
    analysis_store.save(analysis, session_id)


def calculate_lyrics_embeddings(session_id: str, audio_path: str) -> Optional[Tensor]:
    # load lyrics and calculate embedding
    lyrics = LyricsLoader.get_lyrics_for_file(session_id, audio_path)
    if lyrics is None:
        return None
    textual_similarity = TextSimilarity()
    embeddings = textual_similarity.calculate_embeddings(lyrics)
    return embeddings
