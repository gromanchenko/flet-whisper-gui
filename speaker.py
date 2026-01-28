import numpy as np
import json
import os
import toml

# Import VAD from consolidated module
try:
    from .vad import SilenceDetector
except ImportError:
    from vad import SilenceDetector


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors using numpy."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Minimum audio quality thresholds
MIN_RMS_THRESHOLD = 0.005  # Reject silent recordings (lowered for sensitive mics)

# VAD segmentation parameters
VAD_MIN_SEGMENT_SEC = 2.0    # Minimum speech segment length for embedding

# Shared VAD instance for segmentation
_vad_instance = None

def _get_vad():
    """Lazy-load VAD instance."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = SilenceDetector(sample_rate=16000, frame_duration_ms=30, mode=3)
    return _vad_instance


def vad_segment(audio_data, sample_rate=16000, min_segment_sec=VAD_MIN_SEGMENT_SEC):
    """
    Segment audio into speech regions using webrtcvad.
    Returns list of (start_sample, end_sample) tuples for speech segments.
    """
    vad = _get_vad()
    return vad.segment_speech(audio_data, min_segment_sec)


class SpeakerVerifier:
    def __init__(self, config_path=None):
        self.enrolled_embeddings = []  # List of numpy arrays
        # Load from config if provided, otherwise use defaults
        self.min_enroll_seconds = 60  # Blueprint: 60s enrollment
        self.strict_mode = True
        self.top_k = 5  # Default K for top-k scoring
        self.enroll_file = "speaker_enrollment.json"
        self.similarity_threshold = 0.75  # Default, can be overridden by config
        self.encoder = None  # Lazy-loaded
        
        if config_path and os.path.exists(config_path):
            try:
                cfg = toml.load(config_path)
                speaker_cfg = cfg.get('speaker', {})
                self.min_enroll_seconds = speaker_cfg.get('min_enroll_seconds', 60)
                self.similarity_threshold = speaker_cfg.get('similarity_threshold', 0.75)
                self.strict_mode = speaker_cfg.get('strict_mode', True)
            except Exception:
                pass  # Use defaults on error 
        
    def load_enrollment(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.enrolled_embeddings = [np.array(e) for e in data.get('embeddings', [])]
                
    def save_enrollment(self, path):
        data = {
            'embeddings': [e.tolist() for e in self.enrolled_embeddings]
        }
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def reset_enrollment(self, path):
        self.enrolled_embeddings = []
        if os.path.exists(path):
            os.remove(path)

    def validate_audio_quality(self, audio_data, sample_rate):
        """
        Validate audio quality for enrollment.
        Returns (is_valid, error_message).
        """
        if audio_data is None or len(audio_data) == 0:
            return False, "No audio data provided"
        
        # Check RMS (volume level)
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < MIN_RMS_THRESHOLD:
            return False, f"Audio too quiet (RMS={rms:.4f}). Please speak louder."
        
        # Check for clipping (too loud)
        if np.max(np.abs(audio_data)) > 0.99:
            return False, "Audio is clipping. Please move away from the microphone."
        
        # Check minimum duration
        duration = len(audio_data) / sample_rate
        if duration < 5:
            return False, f"Recording too short ({duration:.1f}s). Need at least 5 seconds."
        
        return True, None

    def enroll_voice(self, audio_data, sample_rate, validate=True):
        """
        Process audio, extract embedding(s), add to enrolled_embeddings.
        Uses VAD to segment audio and create multiple embeddings for robustness.
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate (should be 16000)
            validate: whether to validate audio quality (default True)
            
        Returns:
            (success, error_message) tuple
        """
        if validate:
            is_valid, error = self.validate_audio_quality(audio_data, sample_rate)
            if not is_valid:
                return False, error
        
        # Use VAD to segment audio into speech regions
        segments = vad_segment(audio_data, sample_rate)
        
        embeddings_added = 0
        for start, end in segments:
            segment_audio = audio_data[start:end]
            embedding = self._extract_embedding(segment_audio, sample_rate)
            if embedding is not None:
                self.enrolled_embeddings.append(embedding)
                embeddings_added += 1
        
        if embeddings_added > 0:
            return True, f"Added {embeddings_added} voice embedding(s)"
        return False, "Failed to extract voice embeddings from speech segments"

    def add_voice_sample(self, audio_data, sample_rate):
        """
        Add additional voice sample(s) to existing enrollment.
        Validates audio quality before adding.
        
        Returns:
            (success, error_message, embeddings_added) tuple
        """
        is_valid, error = self.validate_audio_quality(audio_data, sample_rate)
        if not is_valid:
            return False, error, 0
        
        initial_count = len(self.enrolled_embeddings)
        
        # Use VAD to segment and add embeddings
        segments = vad_segment(audio_data, sample_rate)
        
        for start, end in segments:
            segment_audio = audio_data[start:end]
            embedding = self._extract_embedding(segment_audio, sample_rate)
            if embedding is not None:
                self.enrolled_embeddings.append(embedding)
        
        added = len(self.enrolled_embeddings) - initial_count
        if added > 0:
            return True, f"Added {added} new embedding(s)", added
        return False, "No valid speech segments found", 0
            
    def verify(self, audio_data, sample_rate):
        """
        Returns True if matches enrolled speaker.
        Uses Min-of-Top-K strategy.
        
        SECURITY: Returns False if no enrollment exists (fail-secure).
        """
        if not self.enrolled_embeddings:
            # FAIL-SECURE: No enrollment means no match
            return False
            
        new_embedding = self._extract_embedding(audio_data, sample_rate)
        if new_embedding is None:
            return False

        # Calculate cosine similarities
        similarities = []
        for enrolled in self.enrolled_embeddings:
            sim = cosine_similarity(new_embedding, enrolled)
            similarities.append(sim)
            
        similarities.sort(reverse=True)
        
        # Take Top-K
        k = min(len(similarities), self.top_k)
        top_k_scores = similarities[:k]
        
        if not top_k_scores:
            return False
            
        # Decision rule: min of top k (conservative)
        score = min(top_k_scores)
        
        return score >= self.similarity_threshold
    
    def get_enrollment_status(self):
        """Return enrollment status info for UI display."""
        return {
            'enrolled': len(self.enrolled_embeddings) > 0,
            'embedding_count': len(self.enrolled_embeddings),
            'threshold': self.similarity_threshold
        }

    def set_threshold(self, threshold):
        """
        Update the similarity threshold.
        
        Args:
            threshold: float between 0.5 and 0.95
        """
        self.similarity_threshold = max(0.5, min(0.95, threshold))

    def _extract_embedding(self, audio_data, sample_rate):
        """
        Extract voice embedding using Resemblyzer.
        
        SECURITY: Fails with RuntimeError if Resemblyzer is unavailable.
        Never returns random data.
        """
        try:
            from resemblyzer import VoiceEncoder
            if self.encoder is None:
                self.encoder = VoiceEncoder()
            
            # Ensure audio is a flat, contiguous 1D numpy array of float32
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.flatten()
            audio = np.ascontiguousarray(audio)
            
            # Resemblyzer expects 16kHz float32 audio
            if len(audio) < 1600:  # Less than 0.1 seconds at 16kHz
                print(f"Audio too short for embedding: {len(audio)} samples")
                return None
                
            emb = self.encoder.embed_utterance(audio)
            return emb
        except ImportError:
            # FAIL-SECURE: Do not fall back to random embeddings
            raise RuntimeError(
                "Speaker verification requires 'resemblyzer' package. "
                "Install with: pip install resemblyzer"
            )
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

