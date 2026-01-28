import pytest
from app import main
from unittest.mock import MagicMock

def test_ui_smoke():
    """
    Test that the UI can be constructed without error.
    """
    try:
        # Mock Page completely
        page = MagicMock()
        page.route = "/"
        page.views = []
        
        # Mock methods to do nothing or simulate behavior
        page.add.side_effect = lambda *args: None
        page.update.side_effect = lambda: None
        page.go.side_effect = lambda route: None
        page.platform = "macos"
        
        # Just verify it doesn't crash
        main(page)
        
    except Exception as e:
        pytest.fail(f"UI construction failed: {e}")
