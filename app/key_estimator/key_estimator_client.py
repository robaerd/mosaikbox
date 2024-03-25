import logging

import numpy as np
from pydub import AudioSegment
import grpc
import app.key_estimator.keyfinder_pb2 as keyfinder_pb2
import app.key_estimator.keyfinder_pb2_grpc as keyfinder_pb2_grpc

logger = logging.getLogger(__name__)


class KeyEstimatorClient:
    def __init__(self):
        # Create a gRPC channel and a stub
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = keyfinder_pb2_grpc.KeyFinderStub(self.channel)

    def estimate_key(self, audio_path: str = None, audio_segment: AudioSegment = None) -> str:
        logger.debug(f"Estimating key for audio {audio_path}")
        if audio_path is None and audio_path is None:
            raise ValueError("Either audio_path or audio_segment must be provided")
        if audio_path is not None:
            audio_segment = AudioSegment.from_file(audio_path)

            # Get the raw audio data (will be in integer form)
        raw_data = np.array(audio_segment.get_array_of_samples())

        # Convert raw audio data to float
        float_data = raw_data.astype(float)

        # Normalize float values between -1 and 1 (PCM audio data)

        pcm_data = float_data / np.iinfo(raw_data.dtype).max

        pcm_data_bytes = pcm_data.astype(np.float32).tobytes()
        logger.debug(f"PCM data size for key estimation request: {pcm_data_bytes}")

        request = keyfinder_pb2.KeyRequest(pcm_data=pcm_data_bytes,
                                           channels=audio_segment.channels,
                                           frame_rate=audio_segment.frame_rate)

        response = self.stub.GetKey(request)
        return response.key
