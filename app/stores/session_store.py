import uuid
from typing import List

from ..exceptions.custom_exceptions import NotFoundException
from ..models.analysis_state import AnalysisState, AnalysisStatus
from ..models.song_schedule import SongScheduleItem


class SessionStore:
    def __init__(self):
        self.active_sessions: dict[str, AnalysisStatus] = {}
        self.song_schedule: dict[str, List[SongScheduleItem]] = {}

    def start_session(self) -> str:
        """
        Start a new upload session. Generates a unique session ID and stores it.
        Returns the session ID.
        """
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = AnalysisStatus(state=AnalysisState.UPLOAD)
        return session_id

    def set_session_status(self, session_id: str, status: AnalysisStatus):
        """
        Set the state of a session.
        """
        self.active_sessions[session_id] = status

    def is_session_active(self, session_id: str) -> bool:
        """
        Check if a session is active.
        """
        return session_id in self.active_sessions

    def get_session_status(self, session_id: str) -> AnalysisStatus:
        """
        Get the state of a session.
        """
        if not self.is_session_active(session_id):
            raise NotFoundException(object_type="Session", object_id=session_id)
        return self.active_sessions[session_id]

    def end_session(self, session_id: str) -> bool:
        """
        End an upload session. Removes the session ID from the active sessions.
        Returns True if the session was successfully ended, False otherwise.
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            if session_id in self.song_schedule:
                del self.song_schedule[session_id]
            return True
        return False

    def get_song_schedule(self, session_id: str) -> List[SongScheduleItem]:
        if session_id not in self.song_schedule:
            raise NotFoundException(object_type="Song Schedule", object_id=session_id)
        return self.song_schedule[session_id]

    def song_schedule_exists(self, session_id: str) -> bool:
        return session_id in self.song_schedule

    def set_song_schedule(self, session_id: str, song_schedule: List[SongScheduleItem]):
        self.song_schedule[session_id] = song_schedule
