import pytest
import sys
from unittest.mock import patch
import toml
import os

# Import from new config module
from config import load_config_dict as load_config

def test_cli_override_whisper(tmp_path):
    """Test that CLI args override config.toml for whisper section"""
    # Mock sys.argv
    test_args = ["app.py", "--model-size", "tiny", "--language", "fr"]
    with patch.object(sys, 'argv', test_args):
        cfg = load_config()
        assert cfg['whisper']['model_size'] == "tiny"
        assert cfg['whisper']['language'] == "fr"

def test_cli_override_vad(tmp_path):
    """Test VAD overrides"""
    test_args = ["app.py", "--min-silence-duration-ms", "999", "--max-speech-duration-s", "5"]
    with patch.object(sys, 'argv', test_args):
        cfg = load_config()
        assert cfg['vad']['min_silence_duration_ms'] == 999
        assert cfg['vad']['max_speech_duration_s'] == 5

def test_cli_override_storage(tmp_path):
    """Test Storage overrides"""
    test_args = ["app.py", "--audio-format", "mp3", "--transcript-dir", "/tmp/t", "--audio-dir", "/tmp/a", "--ffmpeg-path", "/bin/ff"]
    with patch.object(sys, 'argv', test_args):
        cfg = load_config()
        assert cfg['storage']['audio_format'] == "mp3"
        assert cfg['storage']['transcript_dir'] == "/tmp/t"
        assert cfg['storage']['audio_dir'] == "/tmp/a"
        assert cfg['storage']['ffmpeg_path'] == "/bin/ff"

def test_cli_override_speaker(tmp_path):
    """Test Speaker overrides (bools and floats)"""
    test_args = ["app.py", "--strict-mode", "false", "--similarity-threshold", "0.99", "--ui-only-my-voice", "true"]
    with patch.object(sys, 'argv', test_args):
        cfg = load_config()
        assert cfg['speaker']['strict_mode'] is False
        assert cfg['speaker']['similarity_threshold'] == 0.99
        assert cfg['speaker']['ui_only_my_voice'] is True

def test_cli_no_override():
    """Test that default config holds if no args"""
    test_args = ["app.py"]
    with patch.object(sys, 'argv', test_args):
        cfg = load_config()
        # Should match what's in config.toml or default
        # Assuming config.toml exists in cwd
        if os.path.exists("config.toml"):
            file_cfg = toml.load("config.toml")
            assert cfg['whisper']['model_size'] == file_cfg['whisper']['model_size']
