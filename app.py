import flet as ft
import queue
import threading
import sys
import time
import atexit
import os
import numpy as np

# Core Logic
from backend import AudioCapture, Transcriber, Storage
from speaker import SpeakerVerifier
from config import load_config_dict, save_config_dict
from ui.enrollment import create_enrollment_dialog, create_add_samples_dialog

# Views
from ui.views.main_view import MainView
from ui.views.settings_view import SettingsView

# Global State
config = load_config_dict()
ENROLLMENT_FILE = "speaker_enrollment.json"

def main(page: ft.Page):
    # App Entry Point
    page.title = "Transcriber"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_icon = "icon.png"
    
    # --- APP CONTEXT (Shared State) ---
    ctx = {
        'capture': None,
        'transcriber': None,
        'storage': None,
        'verifier': SpeakerVerifier(),
        'is_recording': False,
        'start_recording_fn': None, # Bound below
        'stop_recording_fn': None   # Bound below
    }
    
    # Load verifier
    ctx['verifier'].load_enrollment(ENROLLMENT_FILE)

    # --- BUSINESS LOGIC (Start/Stop) ---
    def start_recording():
        if ctx['is_recording']: return
        
        ctx['is_recording'] = True
        
        # Init components
        ctx['capture'] = AudioCapture()

        
        # Pubsub injector for backend status
        def status_injector(s):
            page.pubsub.send_all(s)
            
        ctx['transcriber'] = Transcriber(config, status_callback=status_injector, speaker_verifier=ctx['verifier'])
        ctx['storage'] = Storage(config)
        
        transcriber_q = queue.Queue()
        
        # Start Backend Threads
        ctx['capture'].start()
        ctx['transcriber'].start(transcriber_q, page.pubsub)
        
        recording_start_time = time.time()
        
        # Audio Plumbing Thread
        def plumbing_loop():
            while ctx['is_recording']:
                try:
                    # Timer
                    elapsed = int(time.time() - recording_start_time)
                    mins, secs = divmod(elapsed, 60)
                    page.pubsub.send_all({'type': 'timer', 'value': f"{mins:02d}:{secs:02d}"})
                    
                    # Audio Data
                    try: 
                        data = ctx['capture'].q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                        
                    # Visualization RMS
                    try:
                        rms = np.sqrt(np.mean(data**2))
                        page.pubsub.send_all({'type': 'audio_level', 'rms': rms})
                    except: pass
                    
                    # Store & Transcribe
                    if ctx['storage']: ctx['storage'].write_audio(data)
                    transcriber_q.put(data)
                    
                except Exception as e:
                    print(f"Plumbing error: {e}")
                    break
        
        threading.Thread(target=plumbing_loop, daemon=True).start()

    def stop_recording():
        if not ctx['is_recording']: return
        ctx['is_recording'] = False
        
        if ctx['capture']: ctx['capture'].stop()
        if ctx['transcriber']: ctx['transcriber'].stop()
        if ctx['storage']: ctx['storage'].close()
        
        ctx['capture'] = None
        ctx['transcriber'] = None
        
    # Bind to context
    ctx['start_recording_fn'] = start_recording
    ctx['stop_recording_fn'] = stop_recording

    # --- PUBSUB HANDLER ---
    ui_lock = threading.Lock()
    def on_pubsub(msg):
        with ui_lock:
            # Dispatch to active view (MainView only needs these)
            if len(page.views) > 0 and isinstance(page.views[-1], MainView):
                view = page.views[-1]
                try:
                    if isinstance(msg, str):
                        view.set_status(msg)
                    elif isinstance(msg, dict):
                        mtype = msg.get('type')
                        if mtype == 'audio_level':
                            # Handle numpy scalar
                            rms = msg['rms']
                            if hasattr(rms, 'item'): rms = rms.item()
                            view.update_volume(rms)
                        elif mtype == 'timer':
                            view.update_timer(msg['value'])
                        else:
                            # Transcription
                            txt = msg.get('text', '')
                            if txt:
                                view.append_text(txt)
                                if ctx['storage']: ctx['storage'].write_transcript(msg)
                    view.update()
                except Exception as e:
                    # Handle view disposed errors gracefully
                    print(f"View update error: {e}")

    page.pubsub.subscribe(on_pubsub)

    # --- NAVIGATION ---
    def nav_settings(e=None):
        page.views.append(SettingsView(
            page, config, 
            on_save=lambda: print("Settings Saved"), 
            on_navigate_home=lambda: page.views.pop() or page.update(),
            speaker_verifier=ctx['verifier']
        ))
        page.update()
        
    def nav_enroll(e=None):
        if ctx['is_recording']:
             page.overlay.append(ft.SnackBar(content=ft.Text("Stop recording first!"), open=True))
             page.update(); return
             
        dlg = create_enrollment_dialog(page, config, ctx['verifier'], ENROLLMENT_FILE)
        page.overlay.append(dlg)
        dlg.open = True
        page.update()

    def nav_add(e=None):
        if ctx['is_recording']:
             page.overlay.append(ft.SnackBar(content=ft.Text("Stop recording first!"), open=True))
             page.update(); return
             
        dlg = create_add_samples_dialog(page, config, ctx['verifier'], ENROLLMENT_FILE)
        page.overlay.append(dlg)
        dlg.open = True
        page.update()

    def nav_forget(e=None):
        ctx['verifier'].reset_enrollment(ENROLLMENT_FILE)
        page.overlay.append(ft.SnackBar(content=ft.Text("Voice Profile Deleted"), open=True))
        page.update()

    # --- ENTRY POINT ---
    view_main = MainView(
        page, config, ctx,
        on_navigate_settings=nav_settings,
        on_train=nav_enroll,
        on_add=nav_add,
        on_forget=nav_forget
    )
    page.views.append(view_main)
    page.update()
    
    # Cleanup
    atexit.register(stop_recording)

if __name__ == "__main__":
    ft.run(main)
