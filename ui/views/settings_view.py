import flet as ft
import os
from config import save_config_dict

class SettingsView(ft.View):
    def __init__(self, page: ft.Page, config: dict, on_save: callable, on_navigate_home: callable, speaker_verifier):
        super().__init__()
        self.route = "/settings"
        self.pg = page # Renamed to avoid collision
        self.config = config
        self.on_save = on_save
        self.on_navigate_home = on_navigate_home
        self.speaker_verifier = speaker_verifier
        
        # Build UI
        self._build_controls()
        
    def _build_controls(self):
        # 1. Models Section
        self.dd_model_size = ft.Dropdown(
            label="Whisper Model Size",
            options=[
                ft.dropdown.Option("tiny"),
                ft.dropdown.Option("base"), 
                ft.dropdown.Option("small"),
                ft.dropdown.Option("medium"),
                ft.dropdown.Option("large-v3"),
            ],
            value=self.config.get('whisper', {}).get('model_size', 'medium'),
            on_select=self.on_change
        )
        
        self.dd_language = ft.Dropdown(
            label="Language",
            options=[
                ft.dropdown.Option("auto", "Auto-Detect"),
                ft.dropdown.Option("en", "English"),
                ft.dropdown.Option("es", "Spanish"),
                ft.dropdown.Option("fr", "French"),
                ft.dropdown.Option("de", "German"),
                ft.dropdown.Option("ja", "Japanese"),
                ft.dropdown.Option("zh", "Chinese"),
                ft.dropdown.Option("ru", "Russian"),
            ],
            value=self.config.get('whisper', {}).get('language', 'auto'),
            on_select=self.on_change
        )
        
        self.dd_task = ft.Dropdown(
            label="Task",
            options=[
                ft.dropdown.Option("transcribe", "Transcribe (Keep original language)"),
                ft.dropdown.Option("translate", "Translate to English"),
            ],
            value=self.config.get('whisper', {}).get('task', 'transcribe'),
            on_select=self.on_change
        )
        
        self.txt_initial_prompt = ft.TextField(
            label="Initial Prompt (Context)",
            multiline=True,
            min_lines=2,
            value=self.config.get('whisper', {}).get('initial_prompt', ''),
            on_change=self.on_change
        )

        self.txt_model_path = ft.TextField(
            label="Custom Model Path (Optional - overrides size)",
            value=self.config.get('whisper', {}).get('model_path', ''),
            on_change=self.on_change
        )

        # 2. VAD Section
        self.txt_vad_min_silence = ft.TextField(label="Min Silence (ms)", value=str(self.config.get('vad', {}).get('min_silence_duration_ms', 500)), on_change=self.on_change, width=150)
        self.txt_vad_max_speech = ft.TextField(label="Max Speech (s)", value=str(self.config.get('vad', {}).get('max_speech_duration_s', 10)), on_change=self.on_change, width=150)
        self.txt_vad_min_speech = ft.TextField(label="Min Speech (ms)", value=str(self.config.get('vad', {}).get('min_speech_duration_ms', 250)), on_change=self.on_change, width=150)
        self.txt_vad_speech_pad = ft.TextField(label="Speech Pad (ms)", value=str(self.config.get('vad', {}).get('speech_pad_ms', 30)), on_change=self.on_change, width=150)
        
        # 3. Storage Section
        self.txt_transcript_dir = ft.TextField(label="Transcript Dir", value=self.config.get('storage', {}).get('transcript_dir', './Transcripts'), on_change=self.on_change)
        self.txt_audio_dir = ft.TextField(label="Audio Dir", value=self.config.get('storage', {}).get('audio_dir', './Audio'), on_change=self.on_change)
        self.txt_ffmpeg_path = ft.TextField(label="FFmpeg Path", value=self.config.get('storage', {}).get('ffmpeg_path', 'ffmpeg'), on_change=self.on_change)
        self.dd_audio_format = ft.Dropdown(
            label="Audio Format",
            options=[
                ft.dropdown.Option("wav"),
                ft.dropdown.Option("mp3"),
            ],
            value=self.config.get('storage', {}).get('audio_format', 'wav'),
            on_select=self.on_change,
            width=100
        )
        self.chk_keep_media = ft.Checkbox(label="Keep Audio Files", value=self.config.get('storage', {}).get('keep_media', False), on_change=self.on_change)
        self.chk_keep_text = ft.Checkbox(label="Keep Transcripts", value=self.config.get('storage', {}).get('keep_text', True), on_change=self.on_change)

        # 4. Speaker Verification
        self.slider_threshold = ft.Slider(
            min=0.5, max=0.95, divisions=45, 
            label="Threshold: {value}", 
            value=float(self.config.get('speaker', {}).get('similarity_threshold', 0.75)),
            on_change=self.on_change
        )
        self.chk_strict_mode = ft.Checkbox(
            label="Strict Mode (Reject unknown)", 
            value=self.config.get('speaker', {}).get('strict_mode', True),
            on_change=self.on_change
        )
        self.txt_min_enroll = ft.TextField(
            label="Min Enrollment (sec)", 
            value=str(self.config.get('speaker', {}).get('min_enroll_seconds', 60)),
            on_change=self.on_change,
            width=150
        )

        # Danger Zone
        btn_clear_history = ft.ElevatedButton("Clear History", on_click=self.on_clear_history, color="white", bgcolor=ft.Colors.RED_700)
        
        # Assemble Layout
        self.padding = 20
        self.scroll = ft.ScrollMode.AUTO
        self.controls = [
            ft.AppBar(
                title=ft.Text("Settings"),
                leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda e: self.on_navigate_home()),
                bgcolor=ft.Colors.SURFACE_VARIANT
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Whisper Model", size=20, weight=ft.FontWeight.BOLD),
                    self.dd_model_size,
                    self.dd_language,
                    self.dd_task,
                    self.txt_initial_prompt,
                    self.txt_model_path,
                    ft.Divider(),
                    
                    ft.Text("VAD (Voice Activity Detection)", size=20, weight=ft.FontWeight.BOLD),
                    ft.Row([self.txt_vad_min_silence, self.txt_vad_max_speech]),
                    ft.Row([self.txt_vad_min_speech, self.txt_vad_speech_pad]),
                    ft.Divider(),
                    
                    ft.Text("Storage", size=20, weight=ft.FontWeight.BOLD),
                    self.txt_transcript_dir,
                    self.txt_audio_dir,
                    ft.Row([self.dd_audio_format, self.txt_ffmpeg_path]),
                    ft.Row([self.chk_keep_media, self.chk_keep_text]),
                    ft.Divider(),
                    
                    ft.Text("Speaker Verification", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text("Similarity Threshold"),
                    self.slider_threshold,
                    self.chk_strict_mode,
                    self.txt_min_enroll,
                    ft.Divider(),
                    
                    ft.Text("Danger Zone", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.RED),
                    btn_clear_history
                ], spacing=20),
                padding=20,
                width=600, # Max width for readability
                alignment=ft.alignment.top_center
            )
        ]

    def on_change(self, e):
        """Save settings to config object and disk."""
        # Whisper
        self.config['whisper']['model_size'] = self.dd_model_size.value
        self.config['whisper']['language'] = self.dd_language.value
        self.config['whisper']['task'] = self.dd_task.value
        self.config['whisper']['initial_prompt'] = self.txt_initial_prompt.value
        self.config['whisper']['model_path'] = self.txt_model_path.value
        
        # VAD
        try:
            self.config['vad']['min_silence_duration_ms'] = int(self.txt_vad_min_silence.value)
            self.config['vad']['max_speech_duration_s'] = int(self.txt_vad_max_speech.value)
            self.config['vad']['min_speech_duration_ms'] = int(self.txt_vad_min_speech.value)
            self.config['vad']['speech_pad_ms'] = int(self.txt_vad_speech_pad.value)
        except ValueError:
            pass # Ignore invalid numbers while typing
            
        # Storage
        self.config['storage']['transcript_dir'] = self.txt_transcript_dir.value
        self.config['storage']['audio_dir'] = self.txt_audio_dir.value
        self.config['storage']['ffmpeg_path'] = self.txt_ffmpeg_path.value
        self.config['storage']['audio_format'] = self.dd_audio_format.value
        self.config['storage']['keep_media'] = self.chk_keep_media.value
        self.config['storage']['keep_text'] = self.chk_keep_text.value
        
        # Speaker
        self.config['speaker']['strict_mode'] = self.chk_strict_mode.value
        self.config['speaker']['similarity_threshold'] = self.slider_threshold.value
        try:
            self.config['speaker']['min_enroll_seconds'] = int(self.txt_min_enroll.value)
        except: pass
        
        # Persist
        save_config_dict(self.config)
        
        # Update Verification Runtime (if verifier is live)
        if self.speaker_verifier:
            self.speaker_verifier.set_threshold(self.slider_threshold.value)
            
        if self.on_save:
            self.on_save()

    def on_clear_history(self, e):
        def close_dlg(e):
             dlg.open = False
             self.pg.update()
             
        def start_delete(e):
            import shutil
            path = os.path.expanduser(self.config['storage']['transcript_dir'])
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path)
            dlg.open = False
            self.pg.overlay.append(ft.SnackBar(content=ft.Text("History cleared"), open=True))
            self.pg.update()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Confirm Deletion"),
            content=ft.Text("This will delete all persistent transcripts."),
            actions=[
                ft.TextButton("Cancel", on_click=close_dlg),
                ft.TextButton("Delete", on_click=start_delete, style=ft.ButtonStyle(color=ft.Colors.RED)),
            ],
        )
        self.pg.overlay.append(dlg)
        dlg.open = True
        self.pg.update()
