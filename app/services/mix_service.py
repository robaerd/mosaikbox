import concurrent
import logging
from functools import partial
from pathlib import Path
import os
from typing import List

from fastapi import File

import app.services.utilities as utilities
import app.services.music_source_separation as mss
import app.config.path_config as path_config

from ..analysis import audio_processing, preconditions
from ..exceptions.custom_exceptions import NotFoundException, TaskCancelledException, ValidationException
from ..mixing import audio_mixing
from ..models.analysis_state import AnalysisState, AnalysisStatus
from ..models.mix import MixInitiateResponse, AnalysisStartRequest, MixStartRequest, MixModel
from .lyrics_loader import LyricsLoader
from ..models.song_schedule import SongScheduleItem
from ..stores.session_store import SessionStore
from .song_schedule_service import SongScheduleService
from ..task_control.task_control import TaskControlManager

logger = logging.getLogger(__name__)


class MixService:
    def __init__(self):
        self.session_store = SessionStore()
        self.lyrics_loader = LyricsLoader()
        self.song_schedule_service = SongScheduleService()
        self.task_control_manager = TaskControlManager()

    def start_session(self):
        session_id = self.session_store.start_session()
        return MixInitiateResponse(session_id=session_id)

    def check_if_task_should_cancel(self, session_id: str):
        if self.task_control_manager.is_task_cancelled(session_id):
            logger.debug(f"Task for session {session_id} was cancelled, aborting...")
            raise TaskCancelledException(f"Task for session {session_id} was cancelled, aborting...")

    def upload_file(self, session_id: str, file: File):
        if not self.session_store.is_session_active(session_id):
            raise NotFoundException(object_type="Session", object_id=session_id)

        uploaded_songs_path = path_config.get_uploaded_songs_path(session_id)
        # make sure song path exists
        Path(uploaded_songs_path).mkdir(parents=True, exist_ok=True)

        file_location = os.path.join(uploaded_songs_path, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}

    def start_analysis(self, session_id: str, analysis_start_request: AnalysisStartRequest):
        def list_all_songs_in_folder_excluding_seed(cand_song_dir, seed_track):
            cand_song_dir_all = utilities.list_all_songs_in_folder(cand_song_dir)
            seed_track_absolute = os.path.abspath(seed_track)  # normalize paths for comparison
            # exclude the seed track from the list
            return [song for song in cand_song_dir_all if
                    os.path.abspath(os.path.join(cand_song_dir, song)) != seed_track_absolute]

        logger.debug(f"Starting analysis for session {session_id}")
        try:
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.STARTED))
            self.task_control_manager.register_task(session_id)

            if not self.session_store.is_session_active(session_id):
                raise NotFoundException(object_type="Session", object_id=session_id)

            uploaded_songs_path = path_config.get_uploaded_songs_path(session_id)
            seed_track = os.path.join(uploaded_songs_path, analysis_start_request.start_song_name)
            if not preconditions.is_song_long_enough(seed_track):
                raise ValidationException(f"Seed track {seed_track} is too short")
            cand_song_list = list_all_songs_in_folder_excluding_seed(uploaded_songs_path, seed_track)
            cand_song_list = preconditions.filter_songs_too_short(cand_song_list)
            all_songs = cand_song_list.copy()
            all_songs.append(seed_track)

            # download lyrics for all songs
            if analysis_start_request.mix_model == MixModel.MOSAIKBOX and analysis_start_request.is_contextual_information_enabled:
                self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.LYRICS_DOWNLOADING))
                for song in all_songs:
                    if self.task_control_manager.is_task_cancelled(session_id):
                        logger.debug(f"Task for session {session_id} was cancelled, aborting...")
                        return
                    self.lyrics_loader.download_lyrics_for_file(session_id=session_id, path=song)

            # separating stems
            if analysis_start_request.mix_model == MixModel.MOSAIKBOX:
                self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.MSS))
                for song in all_songs:
                    self.check_if_task_should_cancel(session_id)
                    mss.separate(session_id, song)

            # run main analysis
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.ANALYSIS))
            cores = os.cpu_count()
            num_workers = int(cores / 2) + 1
            logger.debug(f"Starting song analysis with {num_workers} workers")
            partial_process_audio = partial(audio_processing.process_audio, session_id=session_id, mix_model=analysis_start_request.mix_model)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                for idx, _ in enumerate(executor.map(partial_process_audio, all_songs)):
                    self.check_if_task_should_cancel(session_id)
                    logger.debug(f'Finished analyzing song {idx + 1}/{len(all_songs)} ({all_songs[idx]})')

            # compute song schedule
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.SCHEDULING))
            self.check_if_task_should_cancel(session_id)
            song_schedule = self.song_schedule_service.compute_song_schedule(session_id, seed_track, cand_song_list,
                                                                             analysis_start_request.is_contextual_information_enabled,
                                                                             mix_model=analysis_start_request.mix_model)
            # save song schedule
            self.song_schedule_service.save_song_schedule(session_id, song_schedule)

            self.session_store.set_song_schedule(session_id, song_schedule)

            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.WAITING_FOR_MIXING))

        except TaskCancelledException as e:
            logger.debug(f"Task for session {session_id} was cancelled, aborting...")
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.CANCELLED))
        except Exception as e:
            logger.error(f"Error while analyzing session {session_id}", exc_info=True)
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.FAILED, error_message=str(e)))

    def start_mixing(self, session_id: str, mix_start_request: MixStartRequest):
        logger.info(f"Starting mix for session {session_id}")
        try:
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.MIXING))
            self.check_if_task_should_cancel(session_id)
            song_schedule = self.get_song_schedule(session_id)

            mix = audio_mixing.mix_songs(session_id=session_id,
                                         song_schedule=song_schedule,
                                         is_drum_removal_enabled=mix_start_request.is_drum_removal_enabled,
                                         is_vocal_removal_enabled=mix_start_request.is_vocal_removal_enabled)
            generated_mix_path = path_config.get_generated_mix_path(session_id)
            # make sure song path exists
            Path(generated_mix_path).mkdir(parents=True, exist_ok=True)

            mix.export(os.path.join(generated_mix_path, "mix.mp3"), format="mp3", bitrate='320k')

            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.COMPLETED))
        except Exception as e:
            logger.error(f"Error while mixing session {session_id}", exc_info=True)
            self.session_store.set_session_status(session_id, AnalysisStatus(state=AnalysisState.FAILED, error_message=str(e)))

    def get_song_schedule(self, session_id: str) -> List[SongScheduleItem]:
        logger.debug(f"Getting song schedule for session {session_id}")
        if self.session_store.song_schedule_exists(session_id):
            return self.session_store.get_song_schedule(session_id)
        else:
            return self.song_schedule_service.load_song_schedule(session_id)

    def get_status(self, session_id: str):
        logger.debug(f"Getting status for session {session_id}")
        return self.session_store.get_session_status(session_id)

    def get_mix_filepath(self, session_id: str) -> str:
        logger.debug(f"Getting mix filepath for session {session_id}")
        mix_filepath = os.path.join(path_config.get_generated_mix_path(session_id), "mix.mp3")
        if not os.path.exists(mix_filepath):
            raise NotFoundException(object_type="Mix", object_id=session_id)
        return mix_filepath

    def end_session(self, session_id: str):
        logger.debug(f"Ending session {session_id}")
        self.task_control_manager.cancel_task(session_id)
        return {"info": f"Session {session_id} ended"}