import pytest
import queue
import numpy as np
import threading
import time
from unittest.mock import MagicMock, patch
from backend import Transcriber

def test_pipeline_gate_integration():
    """
    Test that Transcriber respects SpeakerVerifier results.
    """
    # 1. Setup
    config = {
        'whisper': {'model_size': 'tiny', 'model_path': ''},
        'vad': {'min_silence_duration_ms': 100, 'max_speech_duration_s': 1},
        'speaker': {'ui_only_my_voice': True} # START with verification enabled
    }
    
    mock_verifier = MagicMock()
    
    # Mock WhisperModel to avoid loading real models
    with patch('backend.WhisperModel') as mock_model_cls, \
         patch('backend.SilenceDetector') as mock_vad_cls:
         
        # Mock VAD to always trigger speech so we push to inference
        mock_vad = mock_vad_cls.return_value
        mock_vad.is_speech.return_value = True
        
        # Mock Whisper to return fixed text
        mock_model_instance = mock_model_cls.return_value
        Seg = type('Segment', (), {'text': "Speech", 'start': 0.0, 'end': 1.0})
        mock_model_instance.transcribe.return_value = ([Seg()], None)

        # Instantiate Transcriber
        transcriber = Transcriber(config, speaker_verifier=mock_verifier)
        # Ensure it's enabled (constructor should read config)
        assert transcriber.verification_enabled is True
        
        # Pipelines
        audio_q = queue.Queue()
        result_q = queue.Queue()
        
        # Start Threads
        transcriber.start(audio_q, result_q)
        
        try:
            # --- SCENARIO 1: Verification FAILS (False) ---
            mock_verifier.verify.return_value = False
            
            # Feed 1 second of audio (force flush via max_speech_duration_s=1 or slightly more)
            # 30ms frames -> ~34 frames for 1 sec
            chunk = np.random.uniform(-0.1, 0.1, 480).astype(np.float32) # 30ms of audio
            for _ in range(40): # 1.2s
                audio_q.put(chunk)
                
            # Wait a bit for processing
            time.sleep(1.0)
            
            # Assert NO result
            assert result_q.qsize() == 0, "Expected no output when verification fails"
            print("Verified: Blocked unauthorized voice.")
            
            # --- SCENARIO 2: Verification PASSES (True) ---
            mock_verifier.verify.return_value = True
            
            # Feed more audio
            for _ in range(40):
                audio_q.put(chunk)
                
            # Wait for processing
            # We should get a result now
            try:
                res = result_q.get(timeout=2.0)
                assert res['text'] == "Speech"
                print("Verified: Transcribed authorized voice.")
            except queue.Empty:
                pytest.fail("Verification passed but no transcript produced")

        finally:
            transcriber.stop()
