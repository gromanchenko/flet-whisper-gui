import warnings

# Filter deprecation warning from webrtcvad using pkg_resources
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import warnings

# Filter deprecation warning from webrtcvad using pkg_resources
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import webrtcvad
import numpy as np

class SilenceDetector:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, mode=3):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_length = int(sample_rate * frame_duration_ms / 1000)
        # WebRTC VAD expects 16-bit PCM
        
    def is_speech(self, audio_frame_float32):
        """
        Check if frame contains speech.
        audio_frame_float32: numpy array of float32 samples [-1, 1]
        """
        # Convert float32 to int16
        audio_int16 = (audio_frame_float32 * 32767).astype(np.int16)
        
        # Ensure correct frame length (padding if necessary, though caller should handle)
        if len(audio_int16) < self.frame_length:
             return False # Too short
             
        # Take just the first frame_length samples if longer
        chunk = audio_int16[:self.frame_length]
        
        return self.vad.is_speech(chunk.tobytes(), self.sample_rate)

    def segment_speech(self, audio_data, min_segment_sec=2.0):
        """
        Segment audio into speech regions using webrtcvad.
        
        Args:
            audio_data: numpy array of float32 samples [-1, 1]
            min_segment_sec: minimum segment duration to keep
            
        Returns:
            List of (start_sample, end_sample) tuples for speech segments.
        """
        min_segment_samples = int(self.sample_rate * min_segment_sec)
        num_frames = len(audio_data) // self.frame_length
        
        if num_frames == 0:
            return [(0, len(audio_data))]
        
        # Classify each frame as speech or not
        speech_frames = []
        for i in range(num_frames):
            start = i * self.frame_length
            end = start + self.frame_length
            frame = audio_data[start:end]
            speech_frames.append(self.is_speech(frame))
        
        # Group consecutive speech frames into segments
        segments = []
        in_speech = False
        segment_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                segment_start = i * self.frame_length
                in_speech = True
            elif not is_speech and in_speech:
                segment_end = i * self.frame_length
                if segment_end - segment_start >= min_segment_samples:
                    segments.append((segment_start, segment_end))
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            segment_end = len(audio_data)
            if segment_end - segment_start >= min_segment_samples:
                segments.append((segment_start, segment_end))
        
        # If no valid segments found, return full audio
        if not segments:
            return [(0, len(audio_data))]
        
        return segments
