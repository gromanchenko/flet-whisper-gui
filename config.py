"""
Configuration management for Transcriber.
Centralizes config loading, validation, and persistence.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import argparse
import os
import toml

CONFIG_PATH = "config.toml"


@dataclass
class WhisperConfig:
    model_size: str = "medium"
    model_path: str = ""
    language: str = "auto"
    task: str = "transcribe"
    initial_prompt: str = ""


@dataclass
class VadConfig:
    min_silence_duration_ms: int = 500
    max_speech_duration_s: int = 10
    min_speech_duration_ms: int = 250
    speech_pad_ms: int = 30


@dataclass  
class StorageConfig:
    transcript_dir: str = "./Transcripts"
    audio_dir: str = "./Audio"
    audio_format: str = "wav"
    ffmpeg_path: str = "ffmpeg"
    keep_media: bool = False
    keep_text: bool = False


@dataclass
class SpeakerConfig:
    strict_mode: bool = True
    min_enroll_seconds: int = 60
    similarity_threshold: float = 0.75
    ui_only_my_voice: bool = False


@dataclass
class AppConfig:
    """Main configuration container."""
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    
    # Runtime flags (not persisted)
    mode: str = "live" # Default mode is now live (headless mic)
    no_gui: bool = False
    config_show: bool = False
    source_audio_file: Optional[str] = None
    dst_text_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to nested dict for serialization."""
        return {
            'whisper': asdict(self.whisper),
            'vad': asdict(self.vad),
            'storage': asdict(self.storage),
            'speaker': asdict(self.speaker),
        }
    
    def get(self, key: str, default=None):
        """Dict-like access for backwards compatibility."""
        if hasattr(self, key):
            val = getattr(self, key)
            if hasattr(val, '__dataclass_fields__'):
                return asdict(val)
            return val
        return default


def _str2bool(v):
    """Parse boolean from string."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_cli_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Transcriber - Local voice transcription")
    
    # [whisper]
    parser.add_argument("--model-size", help="Whisper model size")
    parser.add_argument("--model-path", help="Path to local model")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--initial-prompt", help="Initial prompt for Whisper context")
    
    # [vad]
    parser.add_argument("--min-silence-duration-ms", type=int, help="VAD min silence (ms)")
    parser.add_argument("--speech-pad-ms", type=int, help="VAD speech pad (ms)")
    parser.add_argument("--max-speech-duration-s", type=int, help="VAD max speech duration (s)")
    parser.add_argument("--min-speech-duration-ms", type=int, help="VAD min speech duration (ms)")

    # [storage]
    parser.add_argument("--transcript-dir", help="Transcript directory")
    parser.add_argument("--audio-dir", help="Audio directory")
    parser.add_argument("--audio-format", help="Audio format (wav/mp3)")
    parser.add_argument("--ffmpeg-path", help="Path to FFmpeg executable")

    # [speaker]
    parser.add_argument("--strict-mode", type=_str2bool, help="Strict speaker verification")
    parser.add_argument("--min-enroll-seconds", type=int, help="Min enrollment seconds")
    parser.add_argument("--similarity-threshold", type=float, help="Speaker similarity threshold")
    parser.add_argument("--ui-only-my-voice", type=_str2bool, help="Only listen to enrolled voice")

    # [execution]
    parser.add_argument("--mode", choices=['gui', 'live', 'batch'], help="Execution mode: gui, live (mic), or batch (file)")
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode (alias for --mode gui)")
    parser.add_argument("--batch", action="store_true", help="Start in Batch mode (alias for --mode batch)")

    # [headless]
    parser.add_argument("--no-gui", action="store_true", help="[Deprecated] Use --mode instead")
    parser.add_argument("--source-audio-file", help="Input audio file for batch mode")
    parser.add_argument("--dst-text-file", help="Output text file for batch mode")
    
    parser.add_argument("--config-show", action="store_true", help="Show configuration and exit")

    args, _ = parser.parse_known_args()
    return args


def load_config(path: str = CONFIG_PATH) -> AppConfig:
    """Load configuration from file and CLI arguments."""
    cfg = AppConfig()
    
    # Load from TOML file
    if os.path.exists(path):
        data = toml.load(path)
        
        # Whisper
        if 'whisper' in data:
            w = data['whisper']
            cfg.whisper = WhisperConfig(
                model_size=w.get('model_size', cfg.whisper.model_size),
                model_path=w.get('model_path', cfg.whisper.model_path),
                language=w.get('language', cfg.whisper.language),
                task=w.get('task', cfg.whisper.task),
                initial_prompt=w.get('initial_prompt', cfg.whisper.initial_prompt),
            )
        
        # VAD
        if 'vad' in data:
            v = data['vad']
            cfg.vad = VadConfig(
                min_silence_duration_ms=v.get('min_silence_duration_ms', cfg.vad.min_silence_duration_ms),
                max_speech_duration_s=v.get('max_speech_duration_s', cfg.vad.max_speech_duration_s),
                min_speech_duration_ms=v.get('min_speech_duration_ms', cfg.vad.min_speech_duration_ms),
                speech_pad_ms=v.get('speech_pad_ms', cfg.vad.speech_pad_ms),
            )
        
        # Storage
        if 'storage' in data:
            s = data['storage']
            cfg.storage = StorageConfig(
                transcript_dir=s.get('transcript_dir', cfg.storage.transcript_dir),
                audio_dir=s.get('audio_dir', cfg.storage.audio_dir),
                audio_format=s.get('audio_format', cfg.storage.audio_format),
                ffmpeg_path=s.get('ffmpeg_path', cfg.storage.ffmpeg_path),
                keep_media=s.get('keep_media', cfg.storage.keep_media),
                keep_text=s.get('keep_text', cfg.storage.keep_text),
            )
        
        # Speaker
        if 'speaker' in data:
            sp = data['speaker']
            cfg.speaker = SpeakerConfig(
                strict_mode=sp.get('strict_mode', cfg.speaker.strict_mode),
                min_enroll_seconds=sp.get('min_enroll_seconds', cfg.speaker.min_enroll_seconds),
                similarity_threshold=sp.get('similarity_threshold', cfg.speaker.similarity_threshold),
                ui_only_my_voice=sp.get('ui_only_my_voice', cfg.speaker.ui_only_my_voice),
            )
    
    # Apply CLI overrides
    try:
        args = _parse_cli_args()
        
        # Runtime flags logic
        # Default is already 'live' from AppConfig init
        
        if args.gui:
            cfg.mode = 'gui'
        elif args.batch:
            cfg.mode = 'batch'
        elif args.mode:
            cfg.mode = args.mode
        elif args.no_gui:
             # Backwards compatibility
            cfg.mode = "batch" if args.source_audio_file else "live"
            cfg.no_gui = True

        if args.source_audio_file:
            cfg.source_audio_file = args.source_audio_file
        if args.dst_text_file:
            cfg.dst_text_file = args.dst_text_file
        if args.config_show:
            cfg.config_show = True
        
        # Whisper overrides
        if args.model_size:
            cfg.whisper.model_size = args.model_size
        if args.model_path:
            cfg.whisper.model_path = args.model_path
        if args.language:
            cfg.whisper.language = args.language
        if args.initial_prompt:
            cfg.whisper.initial_prompt = args.initial_prompt
        # Note: task is config-only (no CLI flag)
        
        # VAD overrides
        if args.min_silence_duration_ms:
            cfg.vad.min_silence_duration_ms = args.min_silence_duration_ms
        if args.speech_pad_ms:
            cfg.vad.speech_pad_ms = args.speech_pad_ms
        if args.max_speech_duration_s:
            cfg.vad.max_speech_duration_s = args.max_speech_duration_s
        if args.min_speech_duration_ms:
            cfg.vad.min_speech_duration_ms = args.min_speech_duration_ms
        
        # Storage overrides
        if args.transcript_dir:
            cfg.storage.transcript_dir = args.transcript_dir
        if args.audio_dir:
            cfg.storage.audio_dir = args.audio_dir
        if args.audio_format:
            cfg.storage.audio_format = args.audio_format
        if args.ffmpeg_path:
            cfg.storage.ffmpeg_path = args.ffmpeg_path
        
        # Speaker overrides
        if args.strict_mode is not None:
            cfg.speaker.strict_mode = args.strict_mode
        if args.min_enroll_seconds:
            cfg.speaker.min_enroll_seconds = args.min_enroll_seconds
        if args.similarity_threshold:
            cfg.speaker.similarity_threshold = args.similarity_threshold
        if args.ui_only_my_voice is not None:
            cfg.speaker.ui_only_my_voice = args.ui_only_my_voice
            
    except Exception as e:
        print(f"Warning: Failed to parse CLI args: {e}")
    
    return cfg


def save_config(cfg: AppConfig, path: str = CONFIG_PATH) -> None:
    """Save configuration to file."""
    with open(path, 'w') as f:
        toml.dump(cfg.to_dict(), f)


def load_config_dict(path: str = CONFIG_PATH) -> dict:
    """
    Load configuration as dict for backwards compatibility.
    Use this during migration from dict-based config.
    """
    cfg = load_config(path)
    result = cfg.to_dict()
    
    # Add runtime flags
    result['mode'] = cfg.mode
    if cfg.no_gui:
        result['no_gui'] = True
    if cfg.config_show:
        result['config_show'] = True
    if cfg.source_audio_file:
        result['source_audio_file'] = cfg.source_audio_file
    if cfg.dst_text_file:
        result['dst_text_file'] = cfg.dst_text_file
    
    return result


def save_config_dict(cfg_dict: dict, path: str = CONFIG_PATH) -> None:
    """Save dict-format configuration to file."""
    # Only save persistent keys
    persistent = {
        'whisper': cfg_dict.get('whisper', {}),
        'vad': cfg_dict.get('vad', {}),
        'storage': cfg_dict.get('storage', {}),
        'speaker': cfg_dict.get('speaker', {}),
    }
    with open(path, 'w') as f:
        toml.dump(persistent, f)
