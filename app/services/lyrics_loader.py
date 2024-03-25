import logging
import os.path
from pathlib import Path
from typing import Optional

import lyricsgenius

from .audio_loader import get_title_artist_of_audio_file
from app.config.config import GENIUS_API_KEY
from app.config.path_config import get_lyrics_path

logger = logging.getLogger(__name__)


class LyricsLoader:
    def __init__(self):
        self.genius = lyricsgenius.Genius(GENIUS_API_KEY)

    def download_lyrics(self, session_id: str, title: str, artist: str):
        logger.debug(f"Downloading lyrics for {artist} - {title}")
        lyrics_path = get_lyrics_path(session_id)
        result = self.genius.search_song(title=title, artist=artist, get_full_info=False)
        if result is not None and result.lyrics is not None:
            # save lyrics to file. folder structure should be artist/title/lyrics.txt
            path_lyrics_dir = Path(os.path.join(lyrics_path, artist, title))
            # assert the path exists
            path_lyrics_dir.mkdir(parents=True, exist_ok=True)
            path_lyrics_file = Path(os.path.join(path_lyrics_dir, "lyrics.txt"))
            # write lyrics to file
            with open(path_lyrics_file, "w") as f:
                f.write(result.lyrics)

    def download_lyrics_for_file(self, session_id: str, path: str):
        title, artist = get_title_artist_of_audio_file(path)
        if title is not None and artist is not None:
            self.download_lyrics(session_id, title, artist)
        else:
            logger.warning(f"Not downloading lyrics for file {path} because title or artist is None")

    @staticmethod
    def get_lyrics(session_id: str, title: str, artist: str) -> Optional[str]:
        lyrics_path = get_lyrics_path(session_id)
        path_lyrics_file = Path(os.path.join(lyrics_path, artist, title, "lyrics.txt"))
        if path_lyrics_file.is_file():
            with open(path_lyrics_file, "r") as f:
                return f.read()
        else:
            logger.warning(f"Lyrics not found for song {artist} - {title}")
            return None

    @staticmethod
    def get_lyrics_for_file(session_id, path: str) -> Optional[str]:
        title, artist = get_title_artist_of_audio_file(path)
        if title is not None and artist is not None:
            return LyricsLoader.get_lyrics(session_id, title, artist)
        else:
            logger.warning(f"Not getting lyrics for file {path} because title or artist is None")
            return None
