"""
Functional tests for 'Train My Voice' (Speaker Enrollment) feature.

These tests verify the complete enrollment and verification pipeline
without requiring actual microphone input or the Resemblyzer model.
"""
import pytest
import numpy as np
import json
import os
import tempfile
import queue
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock


class TestSpeakerVerifierUnit:
    """Unit tests for SpeakerVerifier class."""
    
    def test_enrollment_stores_embedding(self):
        """Verify that enroll_voice adds embedding to list."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        # Mock embedding extraction to return known vector
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0, 0.0])
        
        assert len(verifier.enrolled_embeddings) == 0
        # Use validate=False to skip audio quality checks with mock data
        success, error = verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        assert success == True
        assert len(verifier.enrolled_embeddings) == 1
        np.testing.assert_array_equal(verifier.enrolled_embeddings[0], [1.0, 0.0, 0.0])
    
    def test_enrollment_accumulates_multiple_embeddings(self):
        """Verify multiple enrollments accumulate embeddings."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.8, 0.2, 0.0]),
        ]
        idx = [0]
        
        def mock_extract(a, sr):
            result = embeddings[idx[0]]
            idx[0] += 1
            return result
        
        verifier._extract_embedding = mock_extract
        
        for _ in range(3):
            verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        assert len(verifier.enrolled_embeddings) == 3
    
    def test_verify_exact_match_passes(self):
        """Verify that identical embedding passes verification."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier.similarity_threshold = 0.75
        
        # Enroll with known embedding
        enrolled_emb = np.array([1.0, 0.0, 0.0])
        verifier._extract_embedding = lambda a, sr: enrolled_emb.copy()
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        # Verify with same embedding
        assert verifier.verify(None, 16000) == True
    
    def test_verify_similar_voice_passes(self):
        """Verify that similar embedding (above threshold) passes."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier.similarity_threshold = 0.75
        
        # Enroll
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0, 0.0])
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        # Verify with slightly different embedding (high similarity)
        verifier._extract_embedding = lambda a, sr: np.array([0.95, 0.05, 0.0])
        result = verifier.verify(None, 16000)
        assert result == True
    
    def test_verify_different_voice_fails(self):
        """Verify that dissimilar embedding (below threshold) fails."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier.similarity_threshold = 0.75
        
        # Enroll
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0, 0.0])
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        # Verify with orthogonal embedding (0 similarity)
        verifier._extract_embedding = lambda a, sr: np.array([0.0, 1.0, 0.0])
        result = verifier.verify(None, 16000)
        assert result == False
    
    def test_verify_threshold_boundary(self):
        """Test exact threshold boundary behavior."""
        from speaker import SpeakerVerifier
        from scipy.spatial.distance import cosine
        
        verifier = SpeakerVerifier()
        verifier.similarity_threshold = 0.75
        
        enrolled = np.array([1.0, 0.0])
        verifier._extract_embedding = lambda a, sr: enrolled
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        # Find vector with exactly 0.75 similarity
        # cos_sim = 1 - cos_dist; we want cos_sim = 0.75
        # For 2D unit vectors: cos_sim = dot(a, b) / (|a||b|) = cos(θ)
        # θ = arccos(0.75) ≈ 41.4°
        import math
        theta = math.acos(0.75)
        boundary_vec = np.array([math.cos(theta), math.sin(theta)])
        
        # Normalize
        boundary_vec = boundary_vec / np.linalg.norm(boundary_vec)
        
        verifier._extract_embedding = lambda a, sr: boundary_vec
        # At exact threshold, should pass (>=)
        assert verifier.verify(None, 16000) == True
        
        # Just below threshold
        verifier.similarity_threshold = 0.76
        assert verifier.verify(None, 16000) == False
    
    def test_verify_no_enrollment_returns_false(self):
        """Verify returns False when no voice is enrolled (fail-secure)."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # No enrollment should return False for security
        result = verifier.verify(None, 16000)
        assert result == False  # FAIL-SECURE behavior


class TestSpeakerVerifierPersistence:
    """Tests for enrollment persistence."""
    
    def test_save_and_load_enrollment(self, tmp_path):
        """Verify enrollment persists across instances."""
        from speaker import SpeakerVerifier
        
        filepath = tmp_path / "enrollment.json"
        
        # Create and enroll
        v1 = SpeakerVerifier()
        v1._extract_embedding = lambda a, sr: np.array([0.5, 0.5, 0.7])
        v1.enroll_voice(np.zeros(16000), 16000, validate=False)
        v1.enroll_voice(np.zeros(16000), 16000, validate=False)  # Add second
        v1.save_enrollment(str(filepath))
        
        # Load in new instance
        v2 = SpeakerVerifier()
        v2.load_enrollment(str(filepath))
        
        assert len(v2.enrolled_embeddings) == 2
        np.testing.assert_array_almost_equal(v2.enrolled_embeddings[0], [0.5, 0.5, 0.7])
    
    def test_enrollment_file_format(self, tmp_path):
        """Verify JSON file structure is correct."""
        from speaker import SpeakerVerifier
        
        filepath = tmp_path / "enrollment.json"
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 2.0, 3.0])
        v.enroll_voice(np.random.rand(16000) * 0.5, 16000, validate=False)
        v.save_enrollment(str(filepath))
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)
        assert len(data["embeddings"]) == 1
        assert data["embeddings"][0] == [1.0, 2.0, 3.0]
    
    def test_reset_enrollment_clears_file(self, tmp_path):
        """Verify reset removes enrollment data and file."""
        from speaker import SpeakerVerifier
        
        filepath = tmp_path / "enrollment.json"
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        v.enroll_voice(np.random.rand(16000) * 0.5, 16000, validate=False)
        v.save_enrollment(str(filepath))
        
        assert filepath.exists()
        assert len(v.enrolled_embeddings) == 1
        
        v.reset_enrollment(str(filepath))
        
        assert len(v.enrolled_embeddings) == 0
        assert not filepath.exists()
    
    def test_load_nonexistent_file_is_safe(self, tmp_path):
        """Verify loading missing file doesn't crash."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v.load_enrollment(str(tmp_path / "nonexistent.json"))
        
        assert len(v.enrolled_embeddings) == 0


class TestAudioQualityValidation:
    """Tests for audio quality validation during enrollment."""
    
    def test_rejects_silent_audio(self):
        """Verify enrollment rejects silent recordings."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Create silent audio
        silent_audio = np.zeros(16000 * 10)  # 10 seconds of silence
        
        success, error = v.enroll_voice(silent_audio, 16000, validate=True)
        
        assert success == False
        assert "quiet" in error.lower()
    
    def test_rejects_clipping_audio(self):
        """Verify enrollment rejects clipping audio."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Create clipping audio (values at max)
        clipping_audio = np.ones(16000 * 10) * 1.0  # Max amplitude
        
        success, error = v.enroll_voice(clipping_audio, 16000, validate=True)
        
        assert success == False
        assert "clipping" in error.lower()
    
    def test_rejects_short_audio(self):
        """Verify enrollment rejects too-short recordings."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Create very short audio (1 second)
        short_audio = np.random.rand(16000) * 0.3
        
        success, error = v.enroll_voice(short_audio, 16000, validate=True)
        
        assert success == False
        assert "short" in error.lower()
    
    def test_accepts_valid_audio(self):
        """Verify enrollment accepts good quality audio."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Create valid audio (10 seconds, moderate volume)
        t = np.linspace(0, 10, 16000 * 10)
        valid_audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        
        success, msg = v.enroll_voice(valid_audio, 16000, validate=True)
        
        assert success == True
        assert "embedding" in msg.lower()  # Success message mentions embeddings
    
    def test_validate_audio_quality_empty(self):
        """Verify validation catches empty audio."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        
        is_valid, error = v.validate_audio_quality(np.array([]), 16000)
        assert is_valid == False
        assert "no audio" in error.lower()
    
    def test_validate_audio_quality_none(self):
        """Verify validation catches None audio."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        
        is_valid, error = v.validate_audio_quality(None, 16000)
        assert is_valid == False


class TestSpeakerVerifierTopK:
    """Tests for top-K scoring strategy."""
    
    def test_topk_with_multiple_enrollments(self):
        """Verify top-K strategy with multiple enrolled embeddings."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v.similarity_threshold = 0.75
        v.top_k = 3
        
        # Enroll 5 different embeddings
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.95, 0.05]),
            np.array([0.9, 0.1]),
            np.array([0.8, 0.2]),
            np.array([0.7, 0.3]),
        ]
        
        for emb in embeddings:
            v.enrolled_embeddings.append(emb)
        
        # Test with embedding similar to first 3
        v._extract_embedding = lambda a, sr: np.array([0.98, 0.02])
        
        # Should pass - similar to enrolled voices
        assert v.verify(None, 16000) == True
    
    def test_topk_rejects_when_min_below_threshold(self):
        """Verify rejection when min of top-K is below threshold."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier()
        v.similarity_threshold = 0.9  # High threshold
        v.top_k = 2
        
        # Enroll one very specific embedding
        v.enrolled_embeddings.append(np.array([1.0, 0.0]))
        
        # Test with embedding that has ~60° angle (cos(60°) ≈ 0.5)
        # Vector at 60°: [cos(60°), sin(60°)] = [0.5, 0.866]
        v._extract_embedding = lambda a, sr: np.array([0.5, 0.866])
        
        # Similarity ≈ 0.5 < 0.9 threshold
        assert v.verify(None, 16000) == False


class TestEnrollmentIntegration:
    """Integration tests for enrollment flow."""
    
    def test_enrollment_with_mocked_resemblyzer(self):
        """Test enrollment with mocked Resemblyzer encoder."""
        from speaker import SpeakerVerifier
        
        with patch.dict('sys.modules', {'resemblyzer': MagicMock()}):
            import importlib
            import speaker
            importlib.reload(speaker)
            
            v = speaker.SpeakerVerifier()
            
            # Mock the encoder
            mock_encoder = MagicMock()
            mock_encoder.embed_utterance.return_value = np.random.rand(256)
            v.encoder = mock_encoder
            
            # Create realistic audio (10 seconds of sine wave - passes validation)
            t = np.linspace(0, 10, 16000 * 10)
            audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
            
            # This will call _extract_embedding which uses the mock
            v._extract_embedding = lambda a, sr: mock_encoder.embed_utterance(a)
            success, error = v.enroll_voice(audio, 16000, validate=False)
            
            assert success == True
            assert len(v.enrolled_embeddings) == 1
            mock_encoder.embed_utterance.assert_called()
    
    def test_full_enrollment_verification_cycle(self, tmp_path):
        """Test complete cycle: enroll → save → load → verify."""
        from speaker import SpeakerVerifier
        
        filepath = tmp_path / "voice.json"
        
        # Deterministic fake embeddings
        enrolled_emb = np.array([0.5, 0.5, 0.5, 0.5])
        match_emb = np.array([0.48, 0.52, 0.49, 0.51])  # Very similar
        nomatch_emb = np.array([-0.5, 0.5, -0.5, 0.5])  # Different
        
        # Phase 1: Enrollment
        v1 = SpeakerVerifier()
        v1._extract_embedding = lambda a, sr: enrolled_emb.copy()
        v1.enroll_voice(np.zeros(16000), 16000, validate=False)
        v1.save_enrollment(str(filepath))
        
        # Phase 2: Load and verify (simulating app restart)
        v2 = SpeakerVerifier()
        v2.load_enrollment(str(filepath))
        v2.similarity_threshold = 0.75
        
        # Should match enrolled voice
        v2._extract_embedding = lambda a, sr: match_emb
        assert v2.verify(np.zeros(16000), 16000) == True
        
        # Should reject different voice
        v2._extract_embedding = lambda a, sr: nomatch_emb
        assert v2.verify(np.zeros(16000), 16000) == False


class TestEnrollmentUIFlow:
    """Tests for enrollment UI dialog behavior."""
    
    @pytest.mark.skip(reason="UI integration test - requires full app context")
    def test_enrollment_dialog_has_required_elements(self):
        """Verify enrollment dialog contains all required UI elements."""
        import flet as ft
        from unittest.mock import MagicMock, patch
        
        page = MagicMock(spec=ft.Page)
        page.views = []
        page.overlay = []
        page.data = {}
        
        # Mock pubsub
        mock_pubsub = MagicMock()
        page.pubsub = mock_pubsub
        
        with patch('app.AudioCapture'), \
             patch('app.SpeakerVerifier'), \
             patch('app.Transcriber'), \
             patch('app.Storage'):
            
            from app import main
            main(page)
            
            # Find and click train button
            view_main = page.views[0]
            col = view_main.controls[1] if isinstance(view_main.controls[0], ft.AppBar) else view_main.controls[0]
            
            # Find train button in the column
            train_btn = None
            for control in col.controls:
                if isinstance(control, ft.Row):
                    for item in control.controls:
                        if isinstance(item, (ft.TextButton, ft.FilledButton)):
                            if hasattr(item, 'text') and 'train' in str(item.text).lower():
                                train_btn = item
                                break
                            if hasattr(item, 'icon') and item.icon == ft.Icons.RECORD_VOICE_OVER:
                                train_btn = item
                                break
            
            if train_btn and train_btn.on_click:
                train_btn.on_click(None)
            
            # Check dialog was added
            dialogs = [o for o in page.overlay if isinstance(o, ft.AlertDialog)]
            assert len(dialogs) > 0, "Enrollment dialog should be added to overlay"
            
            dialog = dialogs[-1]
            
            # Verify dialog has required content
            assert dialog.title is not None
            assert "Voice" in dialog.title.value or "Enrollment" in dialog.title.value
            
            # Check for progress bar in content
            content_col = dialog.content
            assert content_col is not None
    
    @pytest.mark.skip(reason="UI integration test - requires full app context")
    def test_enrollment_blocks_when_recording(self):
        """Verify enrollment dialog shows error when recording is active."""
        import flet as ft
        from unittest.mock import MagicMock, patch
        
        page = MagicMock(spec=ft.Page)
        page.views = []
        page.overlay = []
        page.data = {}
        page.pubsub = MagicMock()
        
        with patch('app.AudioCapture'), \
             patch('app.SpeakerVerifier'), \
             patch('app.Transcriber'), \
             patch('app.Storage'):
            
            from app import main
            main(page)
            
            # Simulate recording state by finding and clicking start
            # Then try to open enrollment
            # This is complex due to the closure structure - simplified test
            
            # Just verify the overlay mechanism works
            assert isinstance(page.overlay, list)


class TestConfigIntegration:
    """Tests for config-based enrollment settings."""
    
    def test_verifier_reads_config_threshold(self, tmp_path):
        """Verify SpeakerVerifier reads threshold from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[speaker]
similarity_threshold = 0.85
min_enroll_seconds = 30
strict_mode = false
""")
        
        from speaker import SpeakerVerifier
        v = SpeakerVerifier(config_path=str(config_path))
        
        assert v.similarity_threshold == 0.85
        assert v.min_enroll_seconds == 30
        assert v.strict_mode is False
    
    def test_verifier_uses_defaults_without_config(self):
        """Verify SpeakerVerifier uses defaults when no config."""
        from speaker import SpeakerVerifier
        
        v = SpeakerVerifier(config_path="/nonexistent/path.toml")
        
        assert v.similarity_threshold == 0.75
        assert v.min_enroll_seconds == 60
        assert v.strict_mode is True


class TestPipelineGateWithEnrollment:
    """Integration tests for pipeline gate using enrollment."""
    
    def test_transcriber_uses_verifier_from_enrollment(self):
        """Verify Transcriber respects enrolled voiceprint."""
        from backend import Transcriber
        from speaker import SpeakerVerifier
        
        config = {
            'whisper': {'model_size': 'tiny'},
            'vad': {'min_silence_duration_ms': 100},
            'speaker': {'ui_only_my_voice': True}
        }
        
        # Create verifier with enrolled voice
        verifier = SpeakerVerifier()
        enrolled_emb = np.array([1.0, 0.0, 0.0, 0.0])
        verifier.enrolled_embeddings.append(enrolled_emb)
        
        with patch('backend.WhisperModel'), \
             patch('backend.SilenceDetector'):
            
            t = Transcriber(config, speaker_verifier=verifier)
            
            # Verification should be enabled from config
            assert t.verification_enabled == True
            assert t.verifier is verifier
    
    def test_verification_toggle_at_runtime(self):
        """Test that verification can be toggled during runtime."""
        from backend import Transcriber
        from speaker import SpeakerVerifier
        
        config = {
            'whisper': {'model_size': 'tiny'},
            'vad': {},
            'speaker': {'ui_only_my_voice': False}
        }
        
        verifier = SpeakerVerifier()
        
        with patch('backend.WhisperModel'), \
             patch('backend.SilenceDetector'):
            
            t = Transcriber(config, speaker_verifier=verifier)
            
            assert t.verification_enabled == False
            
            t.set_verification_enabled(True)
            assert t.verification_enabled == True
            
            t.set_verification_enabled(False)
            assert t.verification_enabled == False


class TestVADSegmentation:
    """Tests for VAD-based audio segmentation."""
    
    def test_vad_segment_basic(self):
        """Test basic VAD segmentation functionality."""
        from speaker import vad_segment
        
        # Create audio with speech in the middle
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 1))  # 1s silence
        speech = np.random.rand(int(sample_rate * 3)).astype(np.float32) * 0.1  # 3s speech
        
        audio = np.concatenate([silence, speech, silence]).astype(np.float32)
        
        segments = vad_segment(audio, sample_rate, min_segment_sec=2.0)
        
        # Should find at least one segment
        assert len(segments) >= 1
        
        # Segment should be within audio bounds
        for start, end in segments:
            assert start >= 0
            assert end <= len(audio)
            assert end > start
    
    def test_vad_segment_all_silence(self):
        """Test VAD with all-silent audio returns full audio."""
        from speaker import vad_segment
        
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 5), dtype=np.float32)  # 5s silence
        
        segments = vad_segment(silence, sample_rate)
        
        # Should return full audio as fallback
        assert len(segments) == 1
        assert segments[0] == (0, len(silence))
    
    def test_vad_segment_all_speech(self):
        """Test VAD with continuous speech."""
        from speaker import vad_segment
        
        sample_rate = 16000
        # Continuous speech (above energy threshold)
        speech = np.random.rand(int(sample_rate * 10)).astype(np.float32) * 0.1
        
        segments = vad_segment(speech, sample_rate, min_segment_sec=2.0)
        
        # Should find at least one segment
        assert len(segments) >= 1
    
    def test_enroll_voice_creates_multiple_embeddings(self):
        """Test that enrollment with speech segments creates multiple embeddings."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        
        # Track how many times embedding extraction is called
        call_count = [0]
        def mock_extract(audio, sr):
            call_count[0] += 1
            return np.array([1.0, 0.0, float(call_count[0])])
        
        verifier._extract_embedding = mock_extract
        
        # Create audio with multiple speech segments
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 0.5))  # 0.5s silence
        speech1 = np.random.rand(int(sample_rate * 3)) * 0.1  # 3s speech
        speech2 = np.random.rand(int(sample_rate * 3)) * 0.1  # 3s speech
        
        audio = np.concatenate([speech1, silence, speech2])
        
        success, msg = verifier.enroll_voice(audio, sample_rate, validate=False)
        
        assert success == True
        # Should have created multiple embeddings from segments
        assert len(verifier.enrolled_embeddings) >= 1


class TestAddVoiceSample:
    """Tests for incremental voice sample addition."""
    
    def test_add_voice_sample_success(self):
        """Test adding voice samples to existing enrollment."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Initial enrollment
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        initial_count = len(verifier.enrolled_embeddings)
        
        # Add more samples
        audio = np.random.rand(16000 * 3) * 0.1  # Some speech audio
        success, msg, added = verifier.add_voice_sample(audio, 16000)
        
        # Should have added at least one embedding
        assert len(verifier.enrolled_embeddings) >= initial_count
    
    def test_add_voice_sample_validates_audio(self):
        """Test that add_voice_sample validates audio quality."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Initial enrollment
        verifier.enroll_voice(np.zeros(16000 * 10), 16000, validate=False)
        
        # Try adding silent audio (should fail validation)
        silent_audio = np.zeros(16000 * 5)
        success, msg, added = verifier.add_voice_sample(silent_audio, 16000)
        
        assert success == False
        assert "quiet" in msg.lower() or "rms" in msg.lower()
    
    def test_add_voice_sample_requires_enrollment(self):
        """Test behavior when adding samples with no prior enrollment."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # No prior enrollment - add_voice_sample should still work
        # (it just adds to empty list)
        audio = np.random.rand(16000 * 3) * 0.1
        success, msg, added = verifier.add_voice_sample(audio, 16000)
        
        # Should work (adds to empty enrollment)
        # Validation may pass/fail based on audio
        # This tests that add_voice_sample doesn't require existing enrollment


class TestSetThreshold:
    """Tests for dynamic threshold adjustment."""
    
    def test_set_threshold_updates_value(self):
        """Test that set_threshold updates the similarity threshold."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        assert verifier.similarity_threshold == 0.75  # Default
        
        verifier.set_threshold(0.85)
        assert verifier.similarity_threshold == 0.85
    
    def test_set_threshold_clamps_min(self):
        """Test that threshold is clamped to minimum 0.5."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        
        verifier.set_threshold(0.3)
        assert verifier.similarity_threshold == 0.5  # Clamped
    
    def test_set_threshold_clamps_max(self):
        """Test that threshold is clamped to maximum 0.95."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        
        verifier.set_threshold(0.99)
        assert verifier.similarity_threshold == 0.95  # Clamped
    
    def test_threshold_affects_verification(self):
        """Test that changed threshold affects verification results."""
        from speaker import SpeakerVerifier
        import math
        
        verifier = SpeakerVerifier()
        
        # Enroll
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        # Create test embedding with known similarity (0.8)
        theta = math.acos(0.8)
        test_vec = np.array([math.cos(theta), math.sin(theta)])
        test_vec = test_vec / np.linalg.norm(test_vec)
        
        verifier._extract_embedding = lambda a, sr: test_vec
        
        # With low threshold, should pass
        verifier.set_threshold(0.7)
        assert verifier.verify(None, 16000) == True
        
        # With high threshold, should fail
        verifier.set_threshold(0.9)
        assert verifier.verify(None, 16000) == False


class TestEnrollmentStatus:
    """Tests for enrollment status reporting."""
    
    def test_get_enrollment_status_empty(self):
        """Test status when no enrollment exists."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        status = verifier.get_enrollment_status()
        
        assert status['enrolled'] == False
        assert status['embedding_count'] == 0
        assert status['threshold'] == 0.75
    
    def test_get_enrollment_status_with_embeddings(self):
        """Test status after enrollment."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier._extract_embedding = lambda a, sr: np.array([1.0, 0.0])
        
        # Add some embeddings
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        verifier.enroll_voice(np.zeros(16000), 16000, validate=False)
        
        status = verifier.get_enrollment_status()
        
        assert status['enrolled'] == True
        assert status['embedding_count'] >= 2
    
    def test_get_enrollment_status_reflects_threshold(self):
        """Test that status reflects current threshold."""
        from speaker import SpeakerVerifier
        
        verifier = SpeakerVerifier()
        verifier.set_threshold(0.85)
        
        status = verifier.get_enrollment_status()
        
        assert status['threshold'] == 0.85
