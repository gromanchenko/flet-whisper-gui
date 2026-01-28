import os
import sys
import queue
import threading
import time
import json
import toml
import sounddevice as sd
import numpy as np
from datetime import datetime
from faster_whisper import WhisperModel
import subprocess
import warnings

# Filter noisy warnings from faster_whisper/numpy during silence processing
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

# Import local modules
try:
    from .vad import SilenceDetector
    from .speaker import SpeakerVerifier
except ImportError:
    from vad import SilenceDetector
    from speaker import SpeakerVerifier

def check_model_cache(model_size):
    """
    Check if the model is locally cached in the Hugging Face cache.
    Returns True if found, False otherwise.
    """
    try:
        home = os.path.expanduser("~")
        cache_dir = os.path.join(home, ".cache/huggingface/hub")
        model_id = f"models--Systran--faster-whisper-{model_size}"
        model_path = os.path.join(cache_dir, model_id, "snapshots")
        
        if not os.path.exists(model_path):
            return False
            
        # Check if there's at least one snapshot directory
        snapshots = os.listdir(model_path)
        return len(snapshots) > 0
    except Exception:
        return False

class AudioCapture:
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.q = queue.Queue()
        self.running = False
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def start(self):
        self.running = True
        self.stream = sd.InputStream(samplerate=self.sample_rate, 
                                     channels=1, 
                                     callback=self.callback,
                                     blocksize=self.frame_size)
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}", file=sys.stderr)
            finally:
                self.stream = None
        # Clear the queue to unblock any waiting read()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except:
                break

    def read(self):
        return self.q.get()

class Storage:
    def __init__(self, config):
        self.config = config
        
        # Check settings first - if not keeping media/text, we might not even need files
        self.keep_media = config.get('storage', {}).get('keep_media', False)
        self.keep_text = config.get('storage', {}).get('keep_text', False)
        
        # Default to local directories ("App's folder") to avoid permissions issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_transcript = os.path.join(base_dir, 'Transcripts')
        default_audio = os.path.join(base_dir, 'Audio')
        
        c_transcript = config.get('storage', {}).get('transcript_dir', default_transcript)
        c_audio = config.get('storage', {}).get('audio_dir', default_audio)
        self.format = config.get('storage', {}).get('audio_format', 'wav').lower()
        
        # Handle relative config paths (like "./Transcripts")
        if c_transcript.startswith("./"):
            c_transcript = os.path.join(base_dir, c_transcript[2:])
            
        if c_audio.startswith("./"):
            c_audio = os.path.join(base_dir, c_audio[2:])
            
        self.transcript_dir = os.path.expanduser(c_transcript)
        self.audio_dir = os.path.expanduser(c_audio)
        
        # Ensure directories exist only if we are going to use them
        if self.keep_text:
            try:
                os.makedirs(self.transcript_dir, exist_ok=True)
            except OSError:
                pass 
        if self.keep_media:
            try:
                 os.makedirs(self.audio_dir, exist_ok=True)
            except OSError:
                 pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename_prefix = f"transcript_{timestamp}"
        
        self.wav_path = None
        self.jsonl_path = None
        self.wav_file = None
        self.sf = None

        if self.keep_media:
            self.wav_path = os.path.join(self.audio_dir, f"{self.filename_prefix}.wav")
            import soundfile as sf
            self.sf = sf
            self.wav_file = self.sf.SoundFile(self.wav_path, mode='w', samplerate=16000, channels=1)
            
        if self.keep_text:
            self.jsonl_path = os.path.join(self.transcript_dir, f"{self.filename_prefix}.jsonl")

    def write_audio(self, data):
        if self.wav_file:
            self.wav_file.write(data)

    def write_transcript(self, segment):
        # Segment is dict: {text, start, end}
        if self.jsonl_path:
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(segment) + "\n")

    def close(self):
        if self.wav_file:
            self.wav_file.close()
            
            # Convert to MP3 if requested and file exists
            if self.format == 'mp3' and self.wav_path and os.path.exists(self.wav_path):
                mp3_path = self.wav_path.replace('.wav', '.mp3')
                # Ffmpeg conversion
                ffmpeg_path = self.config.get('storage', {}).get('ffmpeg_path', 'ffmpeg')
                try:
                    subprocess.run([ffmpeg_path, '-y', '-i', self.wav_path, '-codec:a', 'libmp3lame', '-qscale:a', '2', mp3_path], 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Delete intermediate WAV
                    if os.path.exists(mp3_path):
                        os.remove(self.wav_path)
                except Exception as e:
                    print(f"Error converting to MP3: {e}")

class Transcriber:
    def __init__(self, config, status_callback=None, speaker_verifier=None):
        self.config = config
        self.model = None
        self.status_callback = status_callback
        self.verifier = speaker_verifier
        self.verification_enabled = config.get('speaker', {}).get('ui_only_my_voice', False)
        self.running = False
        self._cancel_loading = False
        
        self.model_path = config.get('whisper', {}).get('model_path', '')
        self.model_size = config.get('whisper', {}).get('model_size', 'base')
        
        # Auto-detect device
        self.device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print("CUDA GPU detected: using 'cuda'", file=sys.stderr)
            elif torch.backends.mps.is_available():
                 # faster-whisper (CTranslate2) doesn't fully support MPS yet, usually falls back to CPU 
                 # or requires specific build. Sticking to CPU for Mac unless explicitly overridden.
                 # self.device = "mps" 
                 print("MPS detected but defaulting to 'cpu' for stability", file=sys.stderr)
        except ImportError:
            pass
            
        # Allow config override
        if config.get('whisper', {}).get('device'):
            self.device = config['whisper']['device']
        
        # Internal queues
        # Cap queue at ~20 items to prevent OOM if inference is slower than real-time
        self.inference_queue = queue.Queue(maxsize=20)
                
        # NOTE: Model loading moved to _inference_loop to prevent blocking UI

    def cancel_load(self):
        """Cancel the model loading process (if stuck in download)."""
        self._cancel_loading = True
        self.update_status("Model load cancelled.")

    def set_verification_enabled(self, enabled):
        """Toggle verification mode at runtime"""
        self.verification_enabled = enabled
        print(f"Speaker verification set to: {enabled}")

    def update_status(self, status):
        if self.status_callback:
            self.status_callback(status)

    def load_model(self):
        self.update_status("loading")
        
        model_path = self.config.get('whisper', {}).get('model_path', '')
        model_size = self.config.get('whisper', {}).get('model_size', 'base')
        
        try:
            model_name = model_path if model_path else model_size
            print(f"Loading model: {model_name}")
            
            self.model = WhisperModel(model_name, device=self.device, compute_type="float32")
            self.update_status("loaded")
            print(f"Loaded model: {model_name}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.update_status("unavailable")
            raise RuntimeError(f"Failed to load model: {e}")


    def _vad_loop(self, audio_queue):
        """
        Producer Thread:
        - Reads raw audio from capture
        - Performs VAD
        - Buffers speech
        - Pushes (audio_buffer, prompt) to inference_queue
        """
        # Advanced VAD-based streaming logic
        try:
            vad = SilenceDetector()
        except Exception:
            # Fallback if VAD fails init (e.g. tests)
            vad = None
        
        # Config Params
        min_silence_ms = self.config.get('vad', {}).get('min_silence_duration_ms', 500)
        max_speech_s = self.config.get('vad', {}).get('max_speech_duration_s', 10)
        initial_prompt = self.config.get('whisper', {}).get('initial_prompt', None)
        
        # State
        speech_buffer = [] # List of numpy arrays
        silence_duration_ms = 0
        current_speech_s = 0.0
        
        frame_ms = 30 # Matches AudioCapture defaults
        
        while self.running:
            try:
                # Get Chunk (30ms)
                data = audio_queue.get(timeout=1)
                flat_data = data.flatten()
                
                # Check VAD
                is_speech = True
                if vad:
                    is_speech = vad.is_speech(flat_data)
                
                if is_speech:
                    silence_duration_ms = 0
                    speech_buffer.append(flat_data)
                    current_speech_s += (frame_ms / 1000.0)
                else:
                    if len(speech_buffer) > 0:
                        # We are inside a phrase, add silence as padding
                        speech_buffer.append(flat_data) 
                        silence_duration_ms += frame_ms
                        current_speech_s += (frame_ms / 1000.0)
                        
                        # Trigger: Silence Threshold Reached
                        if silence_duration_ms >= min_silence_ms:
                            self._push_to_inference(speech_buffer, initial_prompt)
                            speech_buffer = []
                            silence_duration_ms = 0
                            current_speech_s = 0.0
                
                # Trigger: Max Duration Reached (Safety split)
                if current_speech_s >= max_speech_s:
                    self._push_to_inference(speech_buffer, initial_prompt)
                    speech_buffer = []
                    silence_duration_ms = 0
                    current_speech_s = 0.0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"VAD loop error: {e}")

    def _push_to_inference(self, buffer_list, prompt):
        if not buffer_list: return
        full_audio = np.concatenate(buffer_list)
        
        # 1. Sanity Check: NaNs or Infs
        if not np.isfinite(full_audio).all():
            return

        # Skip extremely short clips
        min_speech_ms = self.config.get('vad', {}).get('min_speech_duration_ms', 250)
        if len(full_audio) < 16000 * (min_speech_ms / 1000.0): return
        
        # 2. Strict Silence Check (prevents faster-whisper divide-by-zero/overflow)
        # Any signal with max amplitude < 0.005 (0.5%) is effectively silence/noise
        # Standard microphone noise floor is often higher, but we want to catch "digital silence" or "near silence"
        if np.max(np.abs(full_audio)) < 0.005: 
             return
        
        self.inference_queue.put({
            'audio': full_audio,
            'prompt': prompt
        })

    def _inference_loop(self, result_target):
        """
        Consumer Thread:
        - Reads buffered speech from inference_queue
        - Checks Speaker Verification (Pipeline Gate)
        - Runs Inference
        - Pushes text to result_target (queue or pubsub)
        """
        if self._cancel_loading:
            self.running = False
            return

        # Load model here (in background thread)
        try:
            self._cancel_loading = False # Reset flag
            self.update_status("Model: Loading/Downloading...")
            print("Starting model load...", file=sys.stderr)
            self.load_model()
            
            if self._cancel_loading:
                 self.model = None
                 self.update_status("Model load cancelled.")
                 return

            self.update_status("Model: Loaded")
        except Exception as e:
            print(f"Model load failed in thread: {e}")
            self.update_status(f"Error: {e}")
            return # Exit thread if model fails? Or retry? For now exit.

        while self.running:
            try:
                task = self.inference_queue.get(timeout=1)
                audio_data = task['audio']
                prompt = task['prompt']

                # --- PIPELINE GATE: Speaker Verification ---
                if self.verification_enabled:
                    if self.verifier:
                        # Verify
                        is_match = self.verifier.verify(audio_data, 16000)
                        if not is_match:
                            print("Voice ignored: Verification failed.")
                            continue # DROP PACKET
                        else:
                            print("Voice verified.")
                    else:
                        print("Warning: verification_enabled but no verifier instance.")

                # --- INFERENCE ---
                try:
                     language = self.config.get('whisper', {}).get('language', 'en')
                     # None means auto-detect, otherwise use specified language
                     lang_param = None if language == 'auto' else language
                     # task: 'transcribe' = output in original language, 'translate' = translate to English
                     task = self.config.get('whisper', {}).get('task', 'transcribe')
                     segments, info = self.model.transcribe(audio_data, beam_size=5, initial_prompt=prompt, language=lang_param, task=task)
                     for segment in segments:
                         res = {
                             "text": segment.text.strip(),
                             "start": segment.start,
                             "end": segment.end
                         }
                         if res["text"]:
                            # Support both queue (CLI) and pubsub (GUI) patterns
                            if hasattr(result_target, 'put'):
                                result_target.put(res)
                            elif hasattr(result_target, 'send_all'):
                                result_target.send_all(res)
                except Exception as e:
                    print(f"Whisper Inference Error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference loop error: {e}")

    def start(self, audio_queue, result_target):
        """
        Start the transcriber threads.
        
        Args:
            audio_queue: Queue for incoming audio frames
            result_target: Either a Queue (for CLI) or pubsub object (for GUI)
        """
        self.running = True
        
        # Start VAD Loop
        self.vad_thread = threading.Thread(target=self._vad_loop, args=(audio_queue,), daemon=True)
        self.vad_thread.start()
        
        # Start Inference Loop
        self.inf_thread = threading.Thread(target=self._inference_loop, args=(result_target,), daemon=True)
        self.inf_thread.start()

    def stop(self):
        self.running = False
        # Clear queues to unblock any waiting threads
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get_nowait()
            except:
                break
        # Use timeout to prevent hanging if thread is stuck
        if hasattr(self, 'vad_thread') and self.vad_thread.is_alive():
            self.vad_thread.join(timeout=1.0)
        if hasattr(self, 'inf_thread') and self.inf_thread.is_alive():
            self.inf_thread.join(timeout=1.0)
        print("Transcriber stopped.", file=sys.stderr)

    def transcribe_file(self, audio_path):
        """
        Transcribe a complete audio file using the loaded model.
        Returns a list of segment dicts: [{'text':..., 'start':..., 'end':...}]
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        initial_prompt = self.config.get('whisper', {}).get('initial_prompt', None)
        language = self.config.get('whisper', {}).get('language', 'en')
        # None means auto-detect, otherwise use specified language
        lang_param = None if language == 'auto' else language
        # task: 'transcribe' = output in original language, 'translate' = translate to English
        task = self.config.get('whisper', {}).get('task', 'transcribe')
        # We use faster-whisper's built-in VAD for file mode as it's optimized for files
        segments, info = self.model.transcribe(audio_path, beam_size=5, initial_prompt=initial_prompt, vad_filter=True, language=lang_param, task=task)
        
        results = []
        for segment in segments:
            results.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })
        return results
