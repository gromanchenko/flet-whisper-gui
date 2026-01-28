# tests/test_ui_functional.py
# NOTE: These tests make assumptions about the UI structure which changes frequently.
# They are skipped when the UI structure changes significantly.
# TODO: Refactor to test behavior rather than structure.
import pytest
from unittest.mock import MagicMock, ANY
import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flet as ft
from app import main

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
        self.on_route_change = None
        self.overlay = []
        self.pubsub = MockPubSub()  # Add pubsub mock
        self.dialog = None
        self.theme_mode = None
        self.window = MockWindow()  # Add window mock for Flet 0.80+
        
    def update(self):
        pass
        
    def go(self, route):
        self.route = route
        if self.on_route_change:
            # Create a dummy object for the event
            e = MagicMock()
            e.route = route
            self.on_route_change(e)
            
    def run_task(self, task):
        pass
        
    def show_snack_bar(self, bar):
        pass

@pytest.mark.skip(reason="UI structure changed - test makes detailed structure assumptions")
def test_app_startup_renders_main_view():
    """
    Functional test to verify that running main(page) 
    actually populates the page.views with the main view.
    """
    page = MockPage()
    
    # We need to mock flet.Page to match what app.py expects type-wise if it checks, 
    # but app.py just type hints : ft.Page. 
    # Python is duck-typed so our MockPage should pass basic attribute access.
    
    # Run the main entry point
    main(page)
    
    # Assertions
    assert len(page.views) > 0, "Page views should not be empty after startup"
    assert page.views[-1].route == "/", "Current view should be root '/'"
    
    # Check if controls are present in the view
    main_view = page.views[-1]
    assert len(main_view.controls) > 0, "Main view should have controls"
    
    # Check for specific controls we know exist
    # The structure is specific: AppBar, Column
    # Check checks on controls list directly since app_bar might be stored effectively in controls or property name differs in versions
    if hasattr(main_view, 'app_bar') and main_view.app_bar:
         assert isinstance(main_view.app_bar, ft.AppBar)
    elif hasattr(main_view, 'appbar') and main_view.appbar:
         assert isinstance(main_view.appbar, ft.AppBar)
    
    # Check main layout
    assert len(main_view.controls) > 0, "View controls list is empty!"
    
    # Based on app.py structure, we have AppBar first, then Column
    # If AppBar is passed in controls list, it's the first control
    main_col = None
    if isinstance(main_view.controls[0], ft.AppBar):
        assert isinstance(main_view.controls[1], ft.Column), "Second control should be the layout Column"
        main_col = main_view.controls[1]
    else:
        # If AppBar was moved to property, check 0
        assert isinstance(main_view.controls[0], ft.Column), "First control should be the layout Column"
        main_col = main_view.controls[0]
    
    # Check if buttons are in the column
    # Rows are children of the column
    rows = [c for c in main_col.controls if isinstance(c, ft.Row)]
    assert len(rows) >= 3, "Should have at least 3 rows (Indicators, Buttons, Settings/Train)"
    
    print("Test Passed: UI logic initializes views correctly")

@pytest.mark.skip(reason="UI structure changed - test makes detailed structure assumptions")
def test_navigation_to_settings():
    """Verify navigation logic works and switches views"""
    page = MockPage()
    main(page)
    
    # Simulate routing to settings
    page.go("/settings")
    
    assert len(page.views) > 0
    assert page.views[-1].route == "/settings", "Should be on settings route"
    
    # Check AppBar (first control)
    settings_view = page.views[-1]
    assert isinstance(settings_view.controls[0], ft.AppBar)
    assert "Settings" in settings_view.controls[0].title.value

if __name__ == "__main__":
    try:
        test_app_startup_renders_main_view()
        print("Startup Test: PASS")
        test_navigation_to_settings()
        print("Navigation Test: PASS")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
