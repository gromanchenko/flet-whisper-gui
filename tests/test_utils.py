import pytest
from unittest.mock import patch, MagicMock
from utils import open_file_externally
import sys

def test_open_file_externally_macos():
    with patch('sys.platform', 'darwin'):
        with patch('subprocess.run') as mock_run:
            with patch('os.path.exists', return_value=True):
                open_file_externally("/path/to/file")
                mock_run.assert_called_with(["open", "/path/to/file"], check=True)

def test_open_file_externally_windows():
    with patch('sys.platform', 'win32'):
        # os.startfile only exists on Windows, so we must set create=True to mock it on Mac/Linux
        with patch('os.startfile', create=True) as mock_start:
            with patch('os.path.exists', return_value=True):
                open_file_externally("C:\\path\\to\\file")
                mock_start.assert_called_with("C:\\path\\to\\file")

def test_open_file_externally_linux():
    with patch('sys.platform', 'linux'):
        with patch('subprocess.run') as mock_run:
            with patch('os.path.exists', return_value=True):
                open_file_externally("/path/to/file")
                mock_run.assert_called_with(["xdg-open", "/path/to/file"], check=True)

def test_open_file_not_exists():
    with patch('os.path.exists', return_value=False):
        with patch('sys.stderr') as mock_stderr:
             open_file_externally("/path/missing")
             # Should just return/print, not crash
