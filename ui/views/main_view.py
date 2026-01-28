import flet as ft
import time
import os
import threading
from utils import open_file_externally

class MainView(ft.View):
    def __init__(self, page: ft.Page, config: dict, app_context: dict, on_navigate_settings: callable, on_train: callable, on_add: callable, on_forget: callable):
        super().__init__()
        self.route = "/"
        self.pg = page # Renamed to avoid collision with ft.Control.page property
        self.config = config
        self.ctx = app_context # Shared context (capture, transcriber, etc)
        self.on_navigate_settings = on_navigate_settings
        self.on_train = on_train
        self.on_add = on_add
        self.on_forget = on_forget
        
        # State
        self.is_recording = False
        
        self._build_controls()
    
    def _create_action_button(self, text, on_click, width=None, height=None):
        """Helper for left panel buttons"""
        return ft.Container(
            content=ft.Text(text, size=12, color="#CCCCCC", text_align=ft.TextAlign.CENTER),
            bgcolor="#2D2D2D",
            border=ft.Border.all(1, "#555555"),
            border_radius=6,
            height=height, width=width,
            padding=ft.Padding.symmetric(vertical=8),
            alignment=ft.Alignment.CENTER,
            on_click=on_click,
            ink=True
        )

    def _build_controls(self):
        # --- LEFT PANEL CONTROLS ---
        self.btn_settings = self._create_action_button("Settings", self.on_navigate_settings, width=260, height=35)
        self.btn_train = self._create_action_button("Train", self.on_train, width=128, height=35)
        self.btn_add = self._create_action_button("+ Add", self.on_add, width=128, height=35)
        self.btn_forget = self._create_action_button("Forget", self.on_forget, width=260, height=35)

        # Record Button
        self.record_text = ft.Text("Start Recording", size=18, weight=ft.FontWeight.W_500, color="#E0E0E0")
        self.record_container = ft.Container(
            content=self.record_text,
            bgcolor="#2D2D2D",
            border=ft.Border.all(1, "#555555"),
            border_radius=12,
            padding=ft.Padding.symmetric(vertical=20),
            width=260,
            alignment=ft.Alignment.CENTER,
            on_click=self.on_record_click,
            ink=True
        )

        # Indicators
        self.anim_indicator = ft.Container(width=12, height=12, bgcolor=ft.Colors.RED, border_radius=6, visible=False)
        self.recording_label = ft.Text("REC", size=12, weight=ft.FontWeight.W_600, visible=False, color=ft.Colors.RED)
        self.recording_timer = ft.Text("00:00", size=14, weight=ft.FontWeight.W_600, visible=False, color="#E0E0E0")
        self.hint_text = ft.Text("Press Space or click button", size=12, color="#555")
        
        # Volume Meter
        self.vol_bar = ft.ProgressBar(width=100, value=0, bgcolor=ft.Colors.GREY_300)
        self.vol_meter_label = ft.Text("Volume", size=11, color="#666", visible=False)

        # Switch: Only My Voice
        self.switch_voice = ft.Switch(
            value=self.config.get('speaker', {}).get('ui_only_my_voice', False),
            active_color="#9AD7FF",
            active_track_color="#4FA3FF",
            on_change=self.on_voice_switch_change
        )

        # Panel Layout
        left_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Image(src="icon.png", width=32, height=32),
                    ft.Text("Transcriber", size=20, weight=ft.FontWeight.BOLD, color="white")
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                ft.Container(height=15),
                self.btn_settings,
                ft.Container(height=8),
                self.record_container,
                ft.Container(height=8),
                ft.Row([self.btn_train, self.btn_add], spacing=4, alignment=ft.MainAxisAlignment.CENTER),
                self.btn_forget,
                ft.Container(height=16),
                ft.Row([self.anim_indicator, self.recording_label, self.recording_timer], alignment=ft.MainAxisAlignment.CENTER, spacing=8),
                self.hint_text,
                ft.Container(height=16),
                self.vol_meter_label,
                self.vol_bar,
                ft.Container(expand=True),
                ft.Row([self.switch_voice, ft.Text("Only my voice", color="#E0E0E0", size=12)], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(height=10),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
            width=300, bgcolor="#2D2D2D",
            border=ft.Border.all(1, "#444444"), border_radius=12, padding=20
        )

        # --- RIGHT PANEL CONTROLS ---
        # Quick Model Dropdown
        self.dd_model_quick = ft.Dropdown(
            options=[ft.dropdown.Option(s) for s in ["tiny", "base", "small", "medium", "large-v3"]],
            value=self.config.get('whisper', {}).get('model_size', 'medium'),
            width=120, text_size=12, height=35, content_padding=ft.Padding.only(left=10, right=10, bottom=5),
            border_color="#555555", bgcolor="#2D2D2D", color="#E0E0E0",
            on_select=self.on_quick_model_change # Flet 0.80.2 Compat
        )
        
        self.txt_transcript = ft.TextField(
            multiline=True, expand=True, read_only=True, text_size=14, 
            value="Ready to transcribe...", bgcolor="black",
            border_radius=8
        )
        
        # Header Buttons
        btn_copy = self._create_action_button("Copy", self.on_copy, width=60)
        btn_clear = self._create_action_button("Clear", self.on_clear, width=60)
        btn_open = self._create_action_button("Open Folder", self.on_open_folder, width=100)

        right_panel = ft.Container(
            content=ft.Column([
                ft.Row([self.dd_model_quick, ft.Container(expand=True), btn_copy, btn_clear, btn_open], spacing=8),
                ft.Container(content=self.txt_transcript, expand=True, border=ft.Border.all(1, "#555555"), border_radius=8, padding=0),
            ]),
            expand=True, bgcolor="#2D2D2D", border=ft.Border.all(1, "#444444"), padding=15, border_radius=12
        )

        # --- BOTTOM BAR ---
        self.pr_loading = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)
        self.btn_cancel = ft.IconButton(ft.Icons.CANCEL, icon_color=ft.Colors.RED_400, visible=False, on_click=self.on_cancel_load)
        self.status_text = ft.Text("Model: Unloaded", size=12, color="#666666")
        self.tokens_text = ft.Text("Tokens: 0", size=12, color="#666666")
        self.model_path_text = ft.Text("", size=10, color="#555555", italic=True)

        bottom_bar = ft.Container(
            content=ft.Row([
                ft.Row([self.pr_loading, self.btn_cancel, self.status_text], spacing=8),
                ft.Container(expand=True),
                self.model_path_text,
                ft.Container(width=20),
                self.tokens_text
            ]),
            padding=ft.Padding.only(left=10, right=10, top=5)
        )

        self.controls = [
            ft.Row([left_panel, right_panel], expand=True, spacing=10),
            bottom_bar
        ]
        self.padding = 10
        self.expand = True

    # --- EVENT HANDLERS ---
    def on_record_click(self, e):
        # Toggle via Context
        if self.ctx.get('is_recording', False):
            self.ctx['stop_recording_fn']()
        else:
            self.ctx['start_recording_fn']()
        
        self.update_ui_state()

    def update_ui_state(self):
        """Reflect global recording state in UI."""
        is_rec = self.ctx.get('is_recording', False)
        
        if is_rec:
            self.record_text.value = "Stop Recording"
            self.record_text.color = "#FFFFFF"
            self.record_container.bgcolor = "#C62828"
            self.anim_indicator.visible = True
            self.recording_label.visible = True
            self.recording_timer.visible = True
            self.vol_meter_label.visible = True
            self.hint_text.visible = False
        else:
            self.record_text.value = "Start Recording"
            self.record_text.color = "#E0E0E0"
            self.record_container.bgcolor = "#2D2D2D"
            self.anim_indicator.visible = False
            self.recording_label.visible = False
            self.recording_timer.visible = False
            self.vol_meter_label.visible = False
            self.hint_text.visible = True
            self.vol_bar.value = 0
            
        self.pg.update()

    def on_voice_switch_change(self, e):
        val = self.switch_voice.value
        self.config['speaker']['ui_only_my_voice'] = val
        if self.ctx.get('transcriber'):
            self.ctx['transcriber'].set_verification_enabled(val)
        self.pg.update()

    def on_quick_model_change(self, e):
        if self.config['whisper'].get('model_path', '').strip():
            return # Ignore if custom path
        self.config['whisper']['model_size'] = self.dd_model_quick.value
        from ...config import save_config_dict
        save_config_dict(self.config)
        self.pg.update()

    def on_copy(self, e):
        self.pg.set_clipboard(self.txt_transcript.value)
        self.pg.overlay.append(ft.SnackBar(content=ft.Text("Copied to clipboard"), open=True))
        self.pg.update()

    def on_clear(self, e):
        self.txt_transcript.value = ""
        self.tokens_text.value = "Tokens: 0"
        self.pg.update()

    def on_open_folder(self, e):
        path = os.path.expanduser(self.config['storage']['transcript_dir'])
        if not os.path.exists(path): os.makedirs(path, exist_ok=True)
        open_file_externally(path)

    def on_cancel_load(self, e):
        if self.ctx.get('transcriber'):
            self.ctx['transcriber'].cancel_load()
        self.status_text.value = "Model: Cancelled"
        self.pr_loading.visible = False
        self.btn_cancel.visible = False
        self.pg.update()

    # --- PUBLIC UPDATE METHODS ---
    def update_volume(self, rms):
        val = min(float(rms) * 10, 1.0)
        self.vol_bar.value = val
        # Rate limit updates if needed, but Flet handles this okay usually.

    def update_timer(self, time_str):
        self.recording_timer.value = time_str
    
    def append_text(self, text):
        self.txt_transcript.value += f"\n{text}"
        # Token count
        tokens = int(len(self.txt_transcript.value.split()) * 1.3)
        self.tokens_text.value = f"Tokens: {tokens}"

    def set_status(self, message):
        if message == "loading":
            self.pr_loading.visible = True
            self.btn_cancel.visible = True
            self.status_text.value = "Model: Loading..."
        elif message == "loaded":
            self.pr_loading.visible = False
            self.btn_cancel.visible = False
            model = self.config['whisper'].get('model_size', 'medium')
            self.status_text.value = f"Model: {model}"
        elif message.startswith("Error"):
            self.pr_loading.visible = False
            self.btn_cancel.visible = False
            self.status_text.value = message
