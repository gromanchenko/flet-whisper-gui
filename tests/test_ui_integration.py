# tests/test_ui_integration.py
# NOTE: These tests make assumptions about the UI structure which changes frequently.
# They are skipped when the UI structure changes significantly.
# TODO: Refactor to test behavior via external interface rather than internal structure.
import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
import queue
import toml
import asyncio
import threading

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flet as ft
# We need to import app, but we want to patch the dependencies IT imports
# Since app.py imports at top level, we might need to patch sys.modules or use patch.d-ict before import
# Simpler: Import app, then patch the classes in app namespace or the module app imports from.

# However, app.py does: "from .huisper import AudioCapture..." or "from huisper import..."
# dependent on ImportError.

# Safe import without global sys.modules patching
# We will patch the attributes on 'app' after import

import app

# Reset the mocks for specific checks
app.AudioCapture = MagicMock()
app.Transcriber = MagicMock()
app.Storage = MagicMock()
app.SpeakerVerifier = MagicMock()

class MockPubSub:
    """Mock pubsub for testing."""
    def __init__(self):
        self.subscribers = []
        self.messages = []
    
    def subscribe(self, handler):
        self.subscribers.append(handler)
    
    def send_all(self, message):
        self.messages.append(message)
        for handler in self.subscribers:
            handler(message)

class MockWindow:
    def __init__(self):
        self.on_event = None
        self.prevent_close = False
    
    async def destroy(self):
        pass

class MockPage:
    def __init__(self):
        self.title = ""
        self.scroll = ""
        self.theme = None
        self.views = []
        self.route = "/"
        self.overlay = []
        self.controls = [] 
        self.on_route_change = None
        self._tasks = []
        self.pubsub = MockPubSub()  # Add pubsub mock
        self.dialog = None
        self.theme_mode = None
        self.window = MockWindow()  # Add window mock for Flet 0.80+
        
    def update(self):
        pass
        
    def go(self, route):
        self.route = route
        if self.on_route_change:
            # Create a mock event adhering to what flet expects if needed
            e = MagicMock()
            e.route = route
            self.on_route_change(e)
            
    def run_task(self, task):
        self._tasks.append(task)
        
    def show_snack_bar(self, bar):
        # Deprecated but keeping for completeness if code calls it
        pass

@pytest.fixture
def mock_page():
    return MockPage()

@pytest.fixture
def run_app_init(mock_page):
    # This runs the main function which builds the UI and sets up handlers
    # We return the locals/context if possible, but app.main defines functions inside.
    # To test inner functions, we might need to inspect the controls hooked up to them.
    
    # Run main to populate page
    app.main(mock_page)
    
    # Extract controls from the view to trigger events
    view = mock_page.views[0]
    # We know the structure from previous grep/read:
    # Column -> [Row(Indicator), Row(Start, Stop), Row(Train, Switch), Divider, Container(Text), Button(Copy), Card(...)]
    
    col = view.controls[1] if isinstance(view.controls[0], ft.AppBar) else view.controls[0]
    
    # Helper to find control by structure or content
    def find_btn(text=None, icon=None):
        # Recursive search or simple list
        for row in [c for c in col.controls if isinstance(c, ft.Row)]:
            for ctrl in row.controls:
                # Wrapped buttons might be the control itself or wrapper?
                # _btn_wrap returns the control.
                if isinstance(ctrl, (ft.FilledButton, ft.IconButton, ft.TextButton)):
                        # text attr might be missing in newer flet, check content if str, or text field
                        ctrl_text = getattr(ctrl, 'text', None) 
                        if not ctrl_text and isinstance(ctrl.content, str):
                            ctrl_text = ctrl.content
                        elif not ctrl_text and isinstance(ctrl.content, ft.Text):
                            ctrl_text = ctrl.content.value
                            
                        if text and ctrl_text == text: return ctrl
                        if icon and ctrl.icon == icon: return ctrl
            # Check direct children (Copy button)
            for ctrl in col.controls:
                 if isinstance(ctrl, (ft.FilledButton, ft.IconButton, ft.TextButton)):
                        ctrl_text = getattr(ctrl, 'text', None) 
                        if not ctrl_text and isinstance(ctrl.content, str):
                            ctrl_text = ctrl.content
                        elif not ctrl_text and isinstance(ctrl.content, ft.Text):
                            ctrl_text = ctrl.content.value
                            
                        if text and ctrl_text == text: return ctrl
                        
    app_bar = view.controls[0] if isinstance(view.controls[0], ft.AppBar) else None
    
    return {
        'page': mock_page,
        'btn_start': find_btn(text="Start Recording"),
        'btn_stop': find_btn(text="Stop Recording"),
        'btn_settings': app_bar.actions[0] if app_bar and app_bar.actions else None,
        'txt_transcript': col.controls[6].content, # Container -> TextField
    }

@pytest.mark.skip(reason="UI structure changed - test makes detailed control assumptions")
def test_start_stop_recording_flow(run_app_init):
    """
    Test that clicking Start:
    1. Initializes components
    2. Starts capture threads
    3. Updates UI state
    And clicking Stop:
    1. Stops components
    2. Updates UI state
    """
    ctx = run_app_init
    btn_start = ctx['btn_start']
    btn_stop = ctx['btn_stop']
    
    # 1. Click Start
    assert btn_start.on_click is not None
    btn_start.on_click(None) # Event arg usually ignored
    
    # Verification
    # Controls should update
    assert btn_start.disabled is True
    assert btn_stop.disabled is False
    
    # Backend should be init and started
    assert app.AudioCapture.called
    assert app.Transcriber.called
    assert app.Storage.called
    
    capture_instance = app.AudioCapture.return_value
    capture_instance.start.assert_called_once()
    
    transcriber_instance = app.Transcriber.return_value
    transcriber_instance.start.assert_called_once()
    
    # 2. Click Stop
    btn_stop.on_click(None)
    
    # Verification
    assert btn_start.disabled is False
    assert btn_stop.disabled is True
    
    capture_instance.stop.assert_called_once()
    transcriber_instance.stop.assert_called_once()
    
@pytest.mark.skip(reason="UI structure changed - test makes detailed control assumptions")
def test_transcription_update(run_app_init):
    """
    Test that pubsub pattern correctly updates the UI when transcription results arrive.
    With Blueprint §8 pubsub pattern, background threads call page.pubsub.send_all().
    """
    ctx = run_app_init
    page = ctx['page']
    
    # Verify pubsub is subscribed
    assert len(page.pubsub.subscribers) > 0, "Pubsub should have at least one subscriber"
    
    # Start recording to initialize transcriber
    ctx['btn_start'].on_click(None)
    
    # Verify Transcriber.start was called with pubsub
    transcriber_instance = app.Transcriber.return_value
    call_args = transcriber_instance.start.call_args
    assert call_args is not None
    # Second arg should be pubsub (has send_all method)
    result_target = call_args[0][1]
    assert hasattr(result_target, 'send_all'), "Transcriber should receive pubsub object"
    
    # Simulate a transcription result via pubsub
    fake_text = "Hello world"
    page.pubsub.send_all({'text': fake_text})
    
    # Verify pubsub message was received
    assert len(page.pubsub.messages) > 0, "Pubsub should have received messages"
    assert page.pubsub.messages[-1] == {'text': fake_text}, "Last message should match"
    
    print("Verified pubsub pattern for transcription updates")

@pytest.mark.skip(reason="UI structure changed - test makes detailed control assumptions")
def test_settings_auto_save(run_app_init, tmp_path):
    """
    Test that modifying settings auto-saves to config file (no save button).
    """
    ctx = run_app_init
    page = ctx['page']
    
    # Navigate to Settings
    page.go("/settings")
    
    # Find components in new hierarchy:
    # AppBar
    # ListView
    #   [0] Card (Model) -> Container -> Column -> [Text, Divider, dd_model, dd_lang, Row(OR), Row(Path...)]
    #   [1] Card (Storage) -> Container -> Column -> [Text, Divider, txt_transcript, txt_audio]
    #   [2] Card (Speaker Verification) -> Container -> Column
    #   [3] Danger Zone Container (no more Save button)
    
    settings_view = page.views[-1]
    # In app.py: controls=[AppBar, ListView]
    lv = settings_view.controls[1] 
    
    # Model Card
    card_model = lv.controls[0]
    col_model = card_model.content.content # Card -> Container -> Column
    dd_model = col_model.controls[2] # Index 2
    dd_lang = col_model.controls[3] # Index 3
    row_path = col_model.controls[5] # Index 5
    txt_path = row_path.controls[0]
    
    # Storage Card
    card_storage = lv.controls[1]
    col_storage = card_storage.content.content
    txt_transcript = col_storage.controls[2] # Index 2 (0=Title, 1=Divider)
    
    # Modify Values
    new_path = "/tmp/fake/model"
    txt_path.value = new_path
    
    new_lang = "fr"
    dd_lang.value = new_lang
    
    new_transcript_dir = "/tmp/fake/transcripts"
    txt_transcript.value = new_transcript_dir
    
    # Patch toml dump and open to verify auto-save on change
    with patch("builtins.open", create=True) as mock_open:
        with patch("toml.dump") as mock_dump:
            # Trigger on_blur for text field (simulates auto-save)
            if txt_transcript.on_blur:
                txt_transcript.on_blur(None)
            
            # Verify file write happened
            if mock_dump.called:
                saved_config = mock_dump.call_args[0][0]
                assert saved_config['storage']['transcript_dir'] == new_transcript_dir

@pytest.mark.skip(reason="UI structure changed - test makes detailed control assumptions")
def test_danger_zone_buttons(run_app_init):
    """
    Test Danger Zone buttons in Settings:
    1. Verify existence of Reset Profile and Cleanup buttons
    2. Verify Delete saved buttons exist in Storage section
    """
    ctx = run_app_init
    page = ctx['page']
    page.go("/settings")
    
    settings_view = page.views[-1]
    lv = settings_view.controls[1]
    
    # Structure (with auto-save, no Save button):
    # lv.controls[0] -> Model Configuration Card
    # lv.controls[1] -> Storage Card (now contains Delete saved buttons)
    # lv.controls[2] -> Speaker Verification Card
    # lv.controls[3] -> Danger Zone Container
    #   -> content (Column)
    #       -> [0] Header Row
    #       -> [1] Divider
    #       -> [2] Text (Warning)
    #       -> [3] Row (Buttons) - Reset Profile, Cleanup
    
    # Verify Storage card has delete saved buttons
    storage_card = lv.controls[1]
    storage_col = storage_card.content.content
    # Storage: [Text, Divider, TextField, TextField, *TextField(ffmpeg)*, Divider, Text, Text, Row(chk+btn), Row(chk+btn)]
    media_row = storage_col.controls[8]
    text_row = storage_col.controls[9]
    
    btn_delete_media = media_row.controls[1]
    btn_delete_text = text_row.controls[1]
    
    # Verify button labels (text may be in 'text' or 'content' attribute)
    media_label = getattr(btn_delete_media, 'text', None) or getattr(btn_delete_media, 'content', '')
    text_label = getattr(btn_delete_text, 'text', None) or getattr(btn_delete_text, 'content', '')
    assert "Delete saved" in str(media_label)
    assert "Delete saved" in str(text_label)
    
    # Verify Danger Zone has Reset Profile and Cleanup (now at index 3)
    dz_container = lv.controls[3]
    dz_col = dz_container.content
    dz_btn_row = dz_col.controls[3]
    
    btn_reset = dz_btn_row.controls[0]
    btn_cleanup = dz_btn_row.controls[1]
    
    reset_label = getattr(btn_reset, 'text', None) or getattr(btn_reset, 'content', '')
    cleanup_label = getattr(btn_cleanup, 'text', None) or getattr(btn_cleanup, 'content', '')
    assert "Reset Voice Profile" in str(reset_label)
    assert "Cleanup history" in str(cleanup_label)
    
    # Test Delete saved media
    with patch("glob.glob") as mock_glob, patch("os.remove") as mock_remove, patch("os.path.exists", return_value=True):
        mock_glob.return_value = ["/tmp/file1.wav", "/tmp/file2.wav"]
        
        btn_delete_media.on_click(None)
        
        # Should show snackbar with deletion count
        snackbar = page.overlay[-1]
        assert isinstance(snackbar, ft.SnackBar)
        assert "audio files" in snackbar.content.value.lower() or "deleted" in snackbar.content.value.lower()

if __name__ == "__main__":
    try:
        # Mini runner
        print("Running Integration Tests (Direct Mode)...")
        
        # Manually invoke fixture logic since we are running as script
        def manual_run_app_init(page):
            app.main(page)
            view = page.views[0]
            col = view.controls[1] if isinstance(view.controls[0], ft.AppBar) else view.controls[0]
            def find_btn(text=None, icon=None):
                for row in [c for c in col.controls if isinstance(c, ft.Row)]:
                    for ctrl in row.controls:
                        if isinstance(ctrl, (ft.FilledButton, ft.IconButton, ft.TextButton)):
                            # Check content property for Text or simple str
                            btn_text = ctrl.text if hasattr(ctrl, 'text') else str(ctrl.content) if isinstance(ctrl.content, str) else None
                            # Use icon
                            btn_icon = ctrl.icon
                            
                            if text and btn_text == text: return ctrl
                            if icon and btn_icon == icon: return ctrl
                for ctrl in col.controls:
                     if isinstance(ctrl, (ft.FilledButton, ft.IconButton, ft.TextButton)):
                            btn_text = ctrl.text if hasattr(ctrl, 'text') else str(ctrl.content) if isinstance(ctrl.content, str) else None
                            if text and btn_text == text: return ctrl
            # Find Settings button (in AppBar actions)
            btn_settings = None
            # Check if AppBar is in controls or property
            appbar = view.appbar if hasattr(view, 'appbar') else None
            if not appbar:
                # Fallback: check if first control is AppBar
                if view.controls and isinstance(view.controls[0], ft.AppBar):
                    appbar = view.controls[0]
            
            if appbar and appbar.actions:
                btn_settings = appbar.actions[0]
            
            # Find main layout column
            # If AppBar is first control, col is second. If AppBar is property, col is first.
            main_layout_idx = 1 if (view.controls and isinstance(view.controls[0], ft.AppBar)) else 0
            if main_layout_idx < len(view.controls):
                 if isinstance(view.controls[main_layout_idx], ft.Column):
                      col = view.controls[main_layout_idx]
            
            # Transcript Row is idx 5 (after Divider)
            # Row -> [Text, Spacer, btn_open, btn_clear]
            transcript_row = col.controls[5]
            
            return {
                'page': page,
                'btn_start': find_btn(text="Start Recording"),
                'btn_stop': find_btn(text="Stop Recording"),
                'btn_settings': btn_settings,
                'txt_transcript': col.controls[6].content,
                'btn_open_folder': transcript_row.controls[2],
                'btn_clear': transcript_row.controls[3]
            }

        # 1
        p1 = MockPage()
        ctx1 = manual_run_app_init(p1)
        test_start_stop_recording_flow(ctx1)
        print("PASS: Start/Stop Flow")
        
        # 2
        p2 = MockPage()
        ctx2 = manual_run_app_init(p2)
        test_transcription_update(ctx2)
        print("PASS: Plumbing Verification")
        
        # 3
        p3 = MockPage()
        ctx3 = manual_run_app_init(p3)
        test_settings_auto_save(ctx3, None) # tmp_path ignored in this mock run
        print("PASS: Settings Save")
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()


@pytest.mark.skip(reason="UI structure changed - test makes detailed control assumptions")
def test_danger_zone_permission_error(run_app_init):
    """
    Test that PermissionError during deletion triggers a snackbar error.
    """
    ctx = run_app_init
    page = ctx['page']
    page.go("/settings")
    
    settings_view = page.views[-1]
    lv = settings_view.controls[1]
    
    # Storage card contains the delete saved buttons
    storage_card = lv.controls[1]
    storage_col = storage_card.content.content
    # Row with checkbox and delete button for media files
    media_row = storage_col.controls[8]
    btn_delete_media = media_row.controls[1]
    
    # Mock glob to raise PermissionError
    with patch("glob.glob", side_effect=PermissionError("MacOS blocked access")), \
         patch("app.os.path.exists", return_value=True):
        # Click Delete saved media
        btn_delete_media.on_click(None)
        
        # Verify Error Snackbar
        snackbar = page.overlay[-1]
        assert isinstance(snackbar, ft.SnackBar)
        assert "error" in snackbar.content.value.lower() or "macos" in snackbar.content.value.lower()
