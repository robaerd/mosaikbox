import ast
import csv
import logging
import os

import numpy as np

from app.services import utilities, audio_loader
from app.config import config, path_config
from app.stores import analysis_store
from ..analysis.AnalysisResult import AnalysisResult
from app.analysis.Segment import Segment
from app.exceptions.custom_exceptions import NoCandidateError, NotFoundException
from app.key_estimator import key_distance
from app.stores.song_cache import SongCache

import app.analysis.structural_segmentation as ss
import app.analysis.rhythmic_similarity as rs
import app.analysis.beats as bts
import app.analysis.mixability as mixability
from ..models.mix import MixModel
from ..models.song_schedule import SongScheduleItem

logger = logging.getLogger(__name__)


class SongScheduleService:

    def __init__(self):
        self.in_memory_song_cache = SongCache(song_limit=20)

    @staticmethod
    def extract_audio_vector_segment(audio_vector: np.ndarray, start_time_seconds: float, end_time_seconds: float,
                                     sr: int) -> np.ndarray:
        start_time_samples = int(start_time_seconds * sr)
        end_time_samples = int(end_time_seconds * sr)
        return audio_vector[start_time_samples:end_time_samples]

    @staticmethod
    def determine_best_segment(analysis_result: AnalysisResult, b_shift: int,
                               is_first_song: bool = False) -> tuple[Segment, float]:
        b_shift_time = analysis_result.beats[b_shift, 0] + (
            analysis_result.time_until_first_beat_stretched if analysis_result.time_until_first_beat_stretched else 0)  # beats should already be shifted to start at 0
        original_b_shift_time = b_shift_time
        cand_segment: Segment = None

        for segment in analysis_result.downbeat_segments:
            if segment.start_time <= b_shift_time < segment.end_time:
                cand_segment = segment
                break

        # fix for last segment (since < instead of <= and last segment is ignored)
        # this workaround is not the best, but was the fastest one to implement
        if cand_segment is None:
            for segment in analysis_result.downbeat_segments:
                if segment.start_time <= b_shift_time <= segment.end_time:
                    cand_segment = segment
                    break

        if cand_segment is None:
            # try to find segment with min tolerance window
            for segment in analysis_result.downbeat_segments:
                if segment.start_time - config.SEGMENT_LESSER_TOLERANCE_TIME <= b_shift_time <= segment.end_time:
                    cand_segment = segment
                    # set b_shift_time to start of segment
                    logger.debug(
                        f"Found segment with tolerance window: {config.SEGMENT_LESSER_TOLERANCE_TIME} for beat shift: {b_shift} with time: {b_shift_time}")
                    b_shift_time = segment.start_time
                    break
            if cand_segment is None:
                # could not find segment, not even with tolerance window
                raise NoCandidateError(f'Could not find segment for beat shift: {b_shift} with time: {b_shift_time}')

        segment_min_length = config.SEGMENT_MIN_LENGTH if not is_first_song else 76.0
        cand_segment = ss.merge_segments_until_min_length(original_b_shift_time, analysis_result.downbeat_segments,
                                                          segment_min_length, is_first_song=is_first_song)

        return cand_segment, original_b_shift_time

    # stretches song if possible
    # extracts audio, beat and percussion segment
    # shifts percussion and beats to start at 0
    @staticmethod
    def stretch_extract_and_shift_audio_beats_percussion(
            analysis: AnalysisResult,
            start_beat_time: float,
            end_beat_time: float,
            target_tempo: float
    ):
        current_tempo = analysis.bpm
        # only time stretch audio, as analysis information should already be time stretched
        stretch_playback_rate = utilities.calculate_playback_rate(target_tempo=target_tempo,
                                                                  current_tempo=current_tempo)
        stretch_percental_difference = abs(stretch_playback_rate - 1)
        if stretch_percental_difference > config.MAX_STRETCH_PERCENTAL_DIFFERENCE:
            logger.debug(
                f"Tempo difference too high with {analysis.bpm}bpm and {analysis.audio_path} with: {target_tempo}bpm")
            raise RuntimeError('Tempo difference too high')

        audio, sr, audio_length = audio_loader.load(path=analysis.audio_path, sr=analysis.sample_rate, mono=True)

        if stretch_playback_rate != 1.0:
            logger.debug(f"Stretching audio only with playback rate factor {stretch_playback_rate}")
            audio = utilities.adjust_tempo(audio, sr, stretch_playback_rate)
            audio_length = utilities.get_audio_length(audio, sr)

        audio = SongScheduleService.extract_audio_vector_segment(audio_vector=audio,
                                                                 start_time_seconds=analysis.time_until_first_beat_stretched,
                                                                 end_time_seconds=audio_length,
                                                                 sr=sr)

        adjusted_segment_start_beat_time = start_beat_time - analysis.time_until_first_beat_stretched
        adjusted_segment_end_beat_time = end_beat_time - analysis.time_until_first_beat_stretched
        segment_percussion = rs.get_subset_of_percussion_timing(quantized_percussion=analysis.quantized_percussion,
                                                                start_time=adjusted_segment_start_beat_time,
                                                                end_time=adjusted_segment_end_beat_time)

        segment_percussion = rs.adjust_beat_timing_to_start_at_0(segment_percussion)

        segment_audio = SongScheduleService.extract_audio_vector_segment(audio_vector=audio,
                                                                         start_time_seconds=adjusted_segment_start_beat_time,
                                                                         end_time_seconds=adjusted_segment_end_beat_time,
                                                                         sr=sr)

        segment_beats = bts.get_subset_of_beats_by_timing(beats=analysis.beats,
                                                          start_time=adjusted_segment_start_beat_time,
                                                          end_time=adjusted_segment_end_beat_time)

        segment_beats = rs.adjust_beat_timing_to_start_at_0(segment_beats)
        return segment_audio, segment_beats, segment_percussion

    def create_beat_compatibility_vector_from_segments(self, segments: list[Segment], tempo: float,
                                                       time_until_first_beat: float, last_beat_time: float):
        beat_duration = 60 / tempo
        current_time = time_until_first_beat

        atol = 1e-10  # to have a bigger tolerance than the error
        rtol = 0.0  # we don't want relative tolerance as we know that the error is in the area of 1e-15

        weight_vector = []
        while current_time <= last_beat_time:
            segment_found = False
            for segment in segments:
                if np.isclose(segment.start_time, current_time, rtol=rtol, atol=atol):
                    weight_vector.append(segment.compatibility)
                    segment_found = True
                    break
            if not segment_found:
                weight_vector.append(0.0)
            current_time += beat_duration
        return weight_vector

    def create_segment_start_compatibility_vector(self, segments: list[Segment], tempo: float,
                                                  time_until_first_beat: float, last_beat_time: float):
        beat_duration = 60 / tempo
        current_time = time_until_first_beat

        atol = 1e-10  # to have a bigger tolerance than the error
        rtol = 0.0  # we don't want relative tolerance as we know that the error is in the area of 1e-15

        weight_vector = []
        while current_time <= last_beat_time:
            segment_found = False
            for segment in segments:
                if np.isclose(segment.start_time, current_time, rtol=rtol, atol=atol):
                    weight_vector.append(1.0)
                    segment_found = True
                    break
            if not segment_found:
                weight_vector.append(0.0)
            current_time += beat_duration
        return weight_vector

    def create_min_length_compatibility_vector(self, tempo: float, time_until_first_beat: float,
                                               last_beat_time: float):  #
        beat_duration = 60 / tempo
        current_time = time_until_first_beat

        weight_vector = []
        max_shift_time = last_beat_time - config.SEGMENT_MIN_LENGTH - config.SEGMENT_MIN_TRANSITION_OFFSET
        while current_time <= last_beat_time:
            if config.SEGMENT_MIN_TRANSITION_OFFSET <= current_time < max_shift_time:
                weight_vector.append(1.0)
            else:
                weight_vector.append(0.0)
            current_time += beat_duration
        return weight_vector

    def load_song_or_time_stretched_song_if_available(
            self,
            session_id: str,
            playback_rate: float,
            target_tempo: float,
            cand_analysis: AnalysisResult
    ) -> tuple[np.ndarray, int, float]:
        time_stretched_audio_path = path_config.get_time_stretched_audio_path(session_id)
        stretched_audio_path = f'{time_stretched_audio_path}/{cand_analysis.filename_with_hash}_stretched_{playback_rate}.npy'
        if os.path.exists(stretched_audio_path) and playback_rate != 1.0:
            cand_sr = cand_analysis.sample_rate
            cand_song, cand_analysis.beats, cand_analysis.target_bpm, cand_audio_length, cand_analysis.quantized_percussion = utilities.adjust_tempo_and_recalculate_beats(
                session_id=session_id,
                audio=None,
                sr=cand_sr,
                playback_rate=playback_rate,
                target_tempo=target_tempo,
                beats=cand_analysis.beats,
                unique_name=cand_analysis.filename_with_hash,
                in_memory_cache=self.in_memory_song_cache,
                quantized_percussion=cand_analysis.quantized_percussion,
                use_cache=True)
        else:
            if self.in_memory_song_cache.contains(stretched_audio_path) and playback_rate == 1.0:
                cand_song = self.in_memory_song_cache.get_song(stretched_audio_path)
                cand_sr = cand_analysis.sample_rate
                cand_audio_length = utilities.get_audio_length(cand_song, cand_sr)
            else:
                cand_song, cand_sr, cand_audio_length = audio_loader.load(path=cand_analysis.audio_path,
                                                                          sr=cand_analysis.sample_rate, mono=True)
            if playback_rate != 1.0:
                cand_song, cand_analysis.beats, cand_analysis.target_bpm, cand_audio_length, cand_analysis.quantized_percussion = utilities.adjust_tempo_and_recalculate_beats(
                    session_id=session_id,
                    audio=cand_song,
                    sr=cand_sr,
                    playback_rate=playback_rate,
                    target_tempo=target_tempo,
                    beats=cand_analysis.beats,
                    unique_name=cand_analysis.filename_with_hash,
                    in_memory_cache=self.in_memory_song_cache,
                    quantized_percussion=cand_analysis.quantized_percussion,
                    use_cache=True)

        return cand_song, cand_sr, cand_audio_length

    def calculate_similarities_from_analysis(
            self,
            session_id: str,
            base_analysis: AnalysisResult,
            base_segment_audio: np.ndarray,
            base_segment_beats: np.ndarray,
            base_segment_percussion: np.ndarray,
            base_target_tempo: float,
            base_target_key: str,
            cand_analysis: AnalysisResult,
            is_contextual_similarity_enabled: bool) -> tuple[
        list[tuple[Segment, float]], np.ndarray]:
        stretch_playback_rate = utilities.calculate_playback_rate(target_tempo=base_target_tempo,
                                                                  current_tempo=cand_analysis.bpm)
        stretch_percental_difference = abs(stretch_playback_rate - 1)
        if stretch_percental_difference > config.MAX_STRETCH_PERCENTAL_DIFFERENCE:
            logger.debug(
                f"Tempo difference too high for {base_analysis.audio_path} with {base_analysis.bpm}bpm and {cand_analysis.audio_path} with: {cand_analysis.bpm}bpm")
            raise RuntimeError('Tempo difference too high')

        cand_song, cand_sr, cand_audio_length = self.load_song_or_time_stretched_song_if_available(
            session_id=session_id,
            playback_rate=stretch_playback_rate,
            target_tempo=base_target_tempo,
            cand_analysis=cand_analysis)

        if stretch_playback_rate != 1.0:
            utilities.time_stretch_segments(cand_analysis.downbeat_segments, stretch_playback_rate)
            if cand_analysis.vocal_segments is not None:
                cand_analysis.vocal_segments = utilities.time_stretch_interval_list_and_return(
                    cand_analysis.vocal_segments,
                    stretch_playback_rate)
        else:
            cand_analysis.target_bpm = cand_analysis.bpm

        # set time_until_first_beat
        cand_analysis.time_until_first_beat_stretched = cand_analysis.beats[0, 0]

        # now extract cand_song starting from the first beat
        # and shift cand_analysis.beats to start at 0 and the percussion as well to start at 0
        cand_song = self.extract_audio_vector_segment(audio_vector=cand_song,
                                                      start_time_seconds=cand_analysis.beats[0, 0],
                                                      end_time_seconds=cand_audio_length,
                                                      sr=cand_sr)
        cand_analysis.quantized_percussion = rs.adjust_beat_timing_to_start_at_0(cand_analysis.quantized_percussion)
        cand_analysis.beats = rs.adjust_beat_timing_to_start_at_0(cand_analysis.beats)

        # calculate similarity for each beat shift
        mash_res = mixability.calc_mixability(session_id=session_id,
                                              audio1_vector=base_segment_audio,
                                              audio2_vector=cand_song,
                                              audio1_percussion=base_segment_percussion,
                                              audio2_percussion=cand_analysis.quantized_percussion,
                                              audio1_beats=base_segment_beats,
                                              audio2_beats=cand_analysis.beats,
                                              audio1_key=base_target_key,
                                              audio2_key=cand_analysis.key,
                                              audio2_beat_sync_timbre=None,
                                              audio1_sr=base_analysis.sample_rate,
                                              audio2_sr=cand_sr,
                                              enable_contextual_similarity=is_contextual_similarity_enabled,
                                              audio1_lyrics_embedding=base_analysis.lyrics_embedding,
                                              audio2_lyrics_embedding=cand_analysis.lyrics_embedding)

        return mash_res, cand_song

    def calc_final_weighted_mixability(self, res_mash_k, segment_weighting_vector, song_min_length_vector,
                                        segment_start_time_weighting_vector):
        # check if all weighting vectors are equally long
        assert len(segment_weighting_vector) == len(song_min_length_vector) == len(segment_start_time_weighting_vector)

        min_res_mash_length = min(len(res_mash_k), len(segment_weighting_vector), len(song_min_length_vector),
                                  len(segment_start_time_weighting_vector))
        if len(res_mash_k) < min_res_mash_length:
            logger.warning(
                f"Segment re-weighting: res_mash_k has fewer elements {len(res_mash_k)} than expected {min_res_mash_length}, some segments will not be processed.")

        res_mash_k_weighted = []
        for i in range(min_res_mash_length):
            new_mash = (res_mash_k[i] * segment_weighting_vector[i] * song_min_length_vector[i] *
                        segment_start_time_weighting_vector[i])
            res_mash_k_weighted.append(new_mash)
        return res_mash_k_weighted

    def compute_song_schedule(self, session_id: str, seed_track: str, cand_song_list: list[str],
                              is_contextual_similarity_enabled: bool, mix_model: MixModel):
        # load the analysis results of the seed track
        seed_analysis = analysis_store.load(utilities.get_filename_with_hash(seed_track), session_id=session_id)
        song_schedule = []

        seed_segment_time, b_shift_time = self.determine_best_segment(seed_analysis, 0, is_first_song=True)
        end_beat_time = seed_segment_time.end_time

        seed_audio, seed_sr, _ = audio_loader.load(seed_analysis.audio_path, sr=seed_analysis.sample_rate)

        seed_analysis.time_until_first_beat_stretched = 0.0  # seed_analysis.beats[0, 0]

        # add seed song to the schedule
        song_schedule_item = SongScheduleItem(
            audio_path=seed_analysis.audio_path,
            mixability=0,
            start_time=0.0,  # seed_analysis.beats[0, 0],
            end_time=float(end_beat_time),
            total_play_time_start=None,
            total_play_time_end=None,
            original_tempo=seed_analysis.bpm,
            target_tempo=seed_analysis.bpm,
            original_key=seed_analysis.key,
            target_key=seed_analysis.key,
            key_shift=0,
            h_contr=0,
            l_contr=0,
            r_contr=0,
            timbral_contr=0,
            segment_factor=0,
            vocal_intervals_time_stretched=seed_analysis.vocal_segments,
            mix_model=mix_model
        )
        song_schedule.append(song_schedule_item)

        # copy so we can re-execute the cell without reloading the songs
        cand_songs = cand_song_list.copy()
        # initially set to seed_analysis
        base_analysis_segment = seed_analysis
        base_analysis_segment.target_bpm = base_analysis_segment.bpm

        while len(cand_songs) > 0:
            logger.debug(f'Number of candidate songs: {len(cand_songs)}')
            song_similarities = []

            # base_start_time = song_schedule[-1].start_time
            base_end_time = song_schedule[-1].end_time
            base_target_tempo = song_schedule[-1].target_tempo
            base_target_key = song_schedule[-1].target_key

            transition_start_time = base_end_time
            transition_end_time = base_end_time + ((60 / base_target_tempo) * 64)
            base_segment_audio, base_segment_beats, base_segment_percussion = SongScheduleService.stretch_extract_and_shift_audio_beats_percussion(
                base_analysis_segment,
                transition_start_time,
                transition_end_time, base_target_tempo)

            for idx, cand_audio_path in enumerate(cand_songs):

                try:
                    logger.debug(f'Processing {idx + 1}/{len(cand_songs)} candidate song: {cand_audio_path}')
                    cand_analysis_data = analysis_store.load(utilities.get_filename_with_hash(cand_audio_path),
                                                             session_id=session_id)

                    segment_similarities, cand_audio_starting_from_first_beat = self.calculate_similarities_from_analysis(
                        session_id=session_id,
                        base_analysis=base_analysis_segment,
                        base_segment_audio=base_segment_audio,
                        base_segment_beats=base_segment_beats,
                        base_segment_percussion=base_segment_percussion,
                        base_target_tempo=base_target_tempo,
                        base_target_key=base_target_key,
                        cand_analysis=cand_analysis_data,
                        is_contextual_similarity_enabled=is_contextual_similarity_enabled)

                    res_mixability, p_shift, key_shift, b_offset, harmonic_sim_k, spectral_balance_k, rhythm_sim_k, _, _, mixability_k = segment_similarities

                    # additional weighting
                    segment_weighting_vector = self.create_beat_compatibility_vector_from_segments(
                        segments=cand_analysis_data.downbeat_segments,
                        tempo=cand_analysis_data.target_bpm,
                        time_until_first_beat=cand_analysis_data.time_until_first_beat_stretched,
                        last_beat_time=cand_analysis_data.beats[-1, 0])
                    segment_start_time_weighting_vector = self.create_segment_start_compatibility_vector(
                        segments=cand_analysis_data.downbeat_segments,
                        tempo=cand_analysis_data.target_bpm,
                        time_until_first_beat=cand_analysis_data.time_until_first_beat_stretched,
                        last_beat_time=cand_analysis_data.beats[-1, 0])
                    song_min_length_vector = self.create_min_length_compatibility_vector(
                        tempo=cand_analysis_data.target_bpm,
                        time_until_first_beat=cand_analysis_data.time_until_first_beat_stretched,
                        last_beat_time=cand_analysis_data.beats[-1, 0])

                    res_mash_k_weighted = self.calc_final_weighted_mixability(mixability_k, segment_weighting_vector,
                                                                               song_min_length_vector,
                                                                               segment_start_time_weighting_vector)
                    # check if all values of son_min_length_vector are 0.0
                    if np.all(np.array(res_mash_k_weighted) == 0.0):
                        logger.warning(f"All values of song_min_length_vector are 0.0, ignoring song {cand_audio_path}")
                        continue

                    # get new beat shift
                    b_offset_segment_weighted = np.argmax(np.array(res_mash_k_weighted))
                    res_mash_weighted_b_offset_weighted = res_mash_k_weighted[b_offset_segment_weighted]
                    logger.debug(f"Old b_offset: {b_offset}, segment weighted b_offset: {b_offset_segment_weighted}")

                    # since we change the beat offset, we need to also adapt the individual feature contributions
                    h_cue = harmonic_sim_k[b_offset_segment_weighted]
                    l_cue = spectral_balance_k[b_offset_segment_weighted]
                    r_cue = rhythm_sim_k[b_offset_segment_weighted]
                    timbre_contr = 0
                    segment_factor = segment_weighting_vector[b_offset_segment_weighted]

                    try:
                        best_segment, b_shift_time = self.determine_best_segment(analysis_result=cand_analysis_data,
                                                                                 b_shift=b_offset_segment_weighted)

                        # beats, downbeats and percussion should already be time stretched
                        similarity_values = (h_cue, l_cue, r_cue, timbre_contr, segment_factor)

                        song_similarities.append((cand_audio_path, best_segment, b_shift_time,
                                                  res_mash_weighted_b_offset_weighted, cand_analysis_data,
                                                  key_shift, similarity_values))
                    except NoCandidateError as e:
                        logger.warning(f'No compatible segment found for: {cand_audio_path}', exc_info=True)
                        continue

                except RuntimeError as e:
                    logger.error(f'Error processing candidate song: {cand_audio_path}', exc_info=True)
                    continue

            if len(song_similarities) == 0:
                # can happen if the tempo of all candidate songs differ so much that time stretch is not feasible
                logger.debug(f'No candidate song found for base song: {base_analysis_segment.audio_path}')
                break

            song_similarities.sort(key=lambda x: x[3], reverse=True)
            best_cand_audio_path = song_similarities[0][0]
            best_cand_segment = song_similarities[0][1]
            best_cand_b_shift_time = song_similarities[0][2]
            best_cand_similarity = song_similarities[0][3]
            base_analysis_segment = song_similarities[0][4]
            best_cand_original_tempo = base_analysis_segment.bpm
            best_cand_target_tempo = base_analysis_segment.target_bpm
            best_cand_vocal_segments_stretched = base_analysis_segment.vocal_segments
            best_cand_key_shift = song_similarities[0][5]

            best_cand_similarities = song_similarities[0][6]
            h_cue, l_cue, r_cue, timbre_contr, segment_factor = best_cand_similarities

            best_cand_original_key = base_analysis_segment.key
            best_cand_target_key = str(best_cand_key_shift)

            # update base_analysis_segment
            song_schedule_item_best_cand = SongScheduleItem(
                audio_path=best_cand_audio_path,
                mixability=float(best_cand_similarity),
                start_time=float(best_cand_b_shift_time),
                end_time=float(best_cand_segment.end_time),
                total_play_time_start=None,
                total_play_time_end=None,
                original_tempo=best_cand_original_tempo,
                target_tempo=best_cand_target_tempo,
                original_key=best_cand_original_key,
                target_key=best_cand_target_key,
                key_shift=int(best_cand_key_shift),
                h_contr=float(h_cue),
                l_contr=float(l_cue),
                r_contr=float(r_cue),
                timbral_contr=float(timbre_contr),
                segment_factor=segment_factor,
                vocal_intervals_time_stretched=best_cand_vocal_segments_stretched,
                mix_model=mix_model
            )
            song_schedule.append(song_schedule_item_best_cand)
            logger.debug(f'Best candidate song: {best_cand_audio_path} with similarity: {best_cand_similarity}')
            cand_songs.remove(best_cand_audio_path)

        logger.debug("Setting end time of last song to audio length")
        last_audio_path = song_schedule[-1].audio_path
        last_audio_length = utilities.get_audio_length_from_file(last_audio_path)
        song_schedule[-1].end_time = last_audio_length

        song_schedule = self.add_total_playtime_to_song_schedule(song_schedule)
        return song_schedule

    def add_total_playtime_to_song_schedule(self, song_schedule: list[SongScheduleItem]):
        total_playtime = 0.0
        for song_schedule_item in song_schedule:
            new_total_playtime = total_playtime + song_schedule_item.end_time - song_schedule_item.start_time
            song_schedule_item.total_play_time_start = total_playtime
            song_schedule_item.total_play_time_end = new_total_playtime
            total_playtime = new_total_playtime
        return song_schedule

    def save_song_schedule(self, session_id: str, song_schedule: list[SongScheduleItem]):
        song_schedule_path = path_config.get_song_schedule_data_path(session_id)
        # make sure song path exists
        os.makedirs(song_schedule_path, exist_ok=True)
        song_schedule_file = f'{song_schedule_path}/song_schedule.csv'
        song_schedule_json = [song_schedule_item.serialize() for song_schedule_item in song_schedule]

        with open(song_schedule_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=song_schedule_json[0].keys())
            writer.writeheader()
            for d in song_schedule_json:
                writer.writerow(d)

    def load_song_schedule(self, session_id) -> list[SongScheduleItem]:
        song_schedule_path = path_config.get_song_schedule_data_path(session_id)
        song_schedule_file = f'{song_schedule_path}/song_schedule.csv'
        song_schedule = []
        if not os.path.exists(song_schedule_file):
            raise NotFoundException(object_type="Song Schedule", object_id=session_id)
        with open(song_schedule_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_schedule_item = SongScheduleItem(
                    audio_path=str(row["audio_path"]),
                    mixability=float(row["mixability"]) if row["mixability"] not in ['None', ''] else None,
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    total_play_time_start=float(row["total_play_time_start"]),
                    total_play_time_end=float(row["total_play_time_end"]),
                    original_tempo=float(row["original_tempo"]),
                    target_tempo=float(row["target_tempo"]),
                    original_key=str(row["original_key"]),
                    target_key=str(row["target_key"]),
                    key_shift=int(row["key_shift"]),
                    h_contr=float(row["h_contr"]),
                    l_contr=float(row["l_contr"]),
                    r_contr=float(row["r_contr"]),
                    timbral_contr=float(row["timbral_contr"]),
                    segment_factor=float(row["segment_factor"]),
                    vocal_intervals_time_stretched=ast.literal_eval(row["vocal_intervals_time_stretched"]),
                    mix_model=MixModel(str(row["mix_model"]))
                )
                song_schedule.append(song_schedule_item)
        return song_schedule
