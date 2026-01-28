from unittest.mock import MagicMock, patch
import app
from backend import Transcriber

def test_cli_batch_flow(tmp_path):
    # Create dummy audio file
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    
    # Config for batch
    cfg = {
        'whisper': {'model_size': 'tiny'},
        'mode': 'batch',
        'source_audio_file': str(audio_path),
        'dst_text_file': str(tmp_path / "out.txt")
    }
    
    with patch('app.Transcriber') as MockTranscriber:
        # Setup mock behavior
        instance = MockTranscriber.return_value
        instance.transcribe_file.return_value = [{'text': 'Hello', 'start': 0, 'end': 1}]
        
        app.run_cli_mode(cfg)
        
        # Verify call
        instance.transcribe_file.assert_called_with(str(audio_path))
        
        # Check output file was written by app logic
        # Wait, app.run_cli_mode does the writing.
        # We need to verify 'open' called? Or check file system if not mocked?
        # The code does: with open(dst, 'w') ...
        # Since we run real app code for writing (only mocked Transcriber), file SHOULD exist.
        
        assert (tmp_path / "out.txt").exists()
        content = (tmp_path / "out.txt").read_text(encoding='utf-8')
        assert "Hello" in content

def test_transcriber_transcribe_file_integration():
    """Unit test for the new transcribe_file method in backend.py"""
    t = Transcriber({'whisper': {'model_size': 'tiny'}})
    # We mock the internal model to avoid loading real weights
    t.model = MagicMock()
    
    # Mock return from faster-whisper
    Seg = type('Segment', (), {'text': " Test File ", 'start': 0.0, 'end': 1.0})
    t.model.transcribe.return_value = ([Seg()], None)
    
    res = t.transcribe_file("dummy.wav")
    
    assert len(res) == 1
    assert res[0]['text'] == "Test File"
    
    # Verify vad_filter=True, language and task were passed
    t.model.transcribe.assert_called_with("dummy.wav", beam_size=5, initial_prompt=None, vad_filter=True, language='en', task='transcribe')

def test_cli_live_flow():
    cfg = {'mode': 'live', 'whisper': {'model_size': 'tiny'}}
    
    with patch('app.Transcriber') as MockTranscriber, \
         patch('app.AudioCapture') as MockCapture, \
         patch('app.time.sleep', side_effect=KeyboardInterrupt):
         
        app.run_cli_mode(cfg)
        
        # Verify starts
        MockCapture.return_value.start.assert_called()
        MockTranscriber.return_value.start.assert_called()
        
        # Verify stops
        MockTranscriber.return_value.stop.assert_called()
        MockCapture.return_value.stop.assert_called()

