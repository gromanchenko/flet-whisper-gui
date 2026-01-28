import pytest
import numpy as np
import queue
from unittest.mock import MagicMock

class FakeCapture:
    def __init__(self, sample_rate=16000):
        self.q = queue.Queue()
        self.running = False
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False
        
    def add_audio_chunk(self):
        # Add 1s of silence or random noise
        chunk = np.zeros(16000, dtype=np.float32)
        self.q.put(chunk)

class DummyTranscriber:
    def __init__(self, config):
        pass
    def start(self, audio_q, result_q):
        pass
    def stop(self):
        pass

@pytest.fixture
def mock_deps(monkeypatch):
    monkeypatch.setattr("backend.AudioCapture", FakeCapture)
    monkeypatch.setattr("backend.Transcriber", DummyTranscriber)
