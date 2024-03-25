import grpc
import keyfinder_pb2
import keyfinder_pb2_grpc
from pydub import AudioSegment
import numpy as np

# Create a gRPC channel and a stub
channel = grpc.insecure_channel('localhost:50051')
stub = keyfinder_pb2_grpc.KeyFinderStub(channel)


# Load audio file
audio = AudioSegment.from_file("../../ck_if_i_could.m4a")

# Get the raw audio data (will be in integer form)
raw_data = np.array(audio.get_array_of_samples())

# Convert raw audio data to float
float_data = raw_data.astype(float)

# Normalize float values between -1 and 1 (PCM audio data)

pcm_data = float_data / np.iinfo(raw_data.dtype).max

pcm_data_bytes = pcm_data.astype(np.float32).tobytes()

request = keyfinder_pb2.KeyRequest(pcm_data=pcm_data_bytes,
                                   channels=audio.channels,
                                   frame_rate=audio.frame_rate)

response = stub.GetKey(request)
print('The key of the song is', response.key)


# # Read the audio file data
# with open('../../ck_if_i_could.wav', 'rb') as f:
#     audio_data = f.read()
#
# # Create a FindKeyRequest message
# request = keyfinder_pb2.KeyRequest(audio_file=audio_data)
#
# # Call the FindKey RPC
# response = stub.GetKey(request)
#
# # Print the key of the song
# print('The key of the song is', response.key)