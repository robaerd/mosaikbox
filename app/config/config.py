import logging
import os

logger = logging.getLogger(__name__)

COMPUTATION_PATH_PROD = "/data/computation"
COMPUTATION_PATH_DEV = "./mosaikbox-data/computation"

SEGMENT_MIN_LENGTH_BEATS = 64
SEGMENT_MIN_LENGTH = 55.0  # seconds

SEGMENT_LESSER_TOLERANCE_TIME = 5.0  # 5 seconds

SEGMENT_MIN_TRANSITION_OFFSET = 13.0 # seconds | time the segment has to be away from it's boundaries so the crossover works

GENIUS_API_KEY = "YOUR_TOKEN"  # todo: add your genius api key here

DEMUCS_MODEL = "htdemucs"

# how many beats should be considered for the rhythm similarity
RHYTHM_SIMILARITY_BEAT_LENGTH = 32

# percussion gets quantized over 12 bins of a beat
RHYTHM_SIMILARITY_QUANTIZATION_OVER_BEATS = 12

MAX_STRETCH_PERCENTAL_DIFFERENCE = 0.08 # 8%


MONGODB_PORT = 27017
MONGODB_DB = "mosaikboxDB"
MONGODB_USER = "mosaikbox"
MONGODB_PASSWORD = "password"
ENV = os.environ.get("ENV")
if ENV == 'prod':
    MONGODB_URL = "mongodb"
else:
    MONGODB_URL = "localhost"

logger.info(f"Using mongodb url: {MONGODB_URL}")