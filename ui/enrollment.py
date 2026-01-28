"""
Enrollment and voice training UI components.
"""
import flet as ft
import asyncio
import time
import queue
import numpy as np
import os
import sys
import subprocess

# Backend imports
try:
    from ..backend import AudioCapture
except ImportError:
    from backend import AudioCapture

def _load_audio_file(path: str, ffmpeg_path: str = "ffmpeg") -> np.ndarray:
    """
    Load audio file and convert to 16kHz mono float32 using ffmpeg.
    """
    try:
        cmd = [
            ffmpeg_path,
            "-i", path,
            "-f", "f32le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
        ]
        # Run ffmpeg
        process = subprocess.run(cmd, capture_output=True, check=True)
        # Parse output
        audio = np.frombuffer(process.stdout, dtype=np.float32)
        return audio
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

def create_enrollment_dialog(page: ft.Page, config: dict, speaker_verifier, enrollment_file: str) -> ft.AlertDialog:
    """
    Create the Voice Enrollment Dialog.
    Includes Microphone recording and File upload.
    """
    enroll_duration = config.get('speaker', {}).get('min_enroll_seconds', 60)
    
    # --- UI Elements ---
    enroll_progress = ft.ProgressBar(width=400, value=0)
    enroll_status = ft.Text("Choose a method to enroll your voice.")
    
    # Mic Tab Elements
    txt_passage = ft.Text(
        "The Rainbow Passage\n\n"
        "When the sunlight strikes raindrops in the air, they act like a prism and form a rainbow. "
        "The rainbow is a division of white light into many beautiful colors. "
        "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. "
        "There is, according to legend, a boiling pot of gold at one end. "
        "People look, but no one ever finds it. "
        "When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.",
        italic=True,
        size=14,
        text_align=ft.TextAlign.JUSTIFY
    )
    viz_bars = ft.ProgressBar(value=0, color=ft.Colors.GREEN, width=400)
    viz_text = ft.Text("Volume: 0%", size=12)
    btn_start_enroll = ft.FilledButton("Start Recording") # Handler attached later

    # File Tab Elements
    txt_file_path = ft.TextField(
        label="Path to Audio File", 
        hint_text="/Users/me/recordings/my_voice.m4a",
        expand=True,
        text_size=12
    )
    btn_process_file = ft.FilledButton("Process File", icon=ft.Icons.UPLOAD) # Handler attached later
    
    # State flags
    enroll_cancelled = [False]
    dialog_ref = [None]

    # --- Handlers ---
    def close_dialog(e):
        enroll_cancelled[0] = True
        if dialog_ref[0]:
            dialog_ref[0].open = False
        page.update()

    # MIC HANDLER
    async def run_mic_enrollment():
        btn_start_enroll.disabled = True
        btn_process_file.disabled = True
        enroll_status.value = f"Recording... Keep talking for {enroll_duration} seconds."
        page.update()

        cap = AudioCapture()
        cap.start()
        audio_chunks = []
        start_time = time.time()

        try:
            while True:
                if enroll_cancelled[0]:
                    enroll_status.value = "Enrollment cancelled."
                    break
                    
                elapsed = time.time() - start_time
                if elapsed >= enroll_duration:
                    break

                try:
                    chunk = cap.q.get_nowait()
                    audio_chunks.append(chunk)
                    
                    # Visualizer
                    rms = np.sqrt(np.mean(chunk**2))
                    viz_val = min(rms * 10, 1.0)
                    viz_bars.value = viz_val
                    viz_text.value = f"Volume: {int(viz_val * 100)}%"

                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                    
                progress = min(elapsed / enroll_duration, 1.0)
                enroll_progress.value = progress
                enroll_status.value = f"Recording... {int(enroll_duration - elapsed)}s remaining"
                page.update()
                await asyncio.sleep(0.01)
                
            cap.stop()
            viz_bars.value = 0
            
            if enroll_cancelled[0]: 
                return

            # Processing
            enroll_status.value = "Processing voice print..."
            enroll_progress.value = None
            page.update()

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks, axis=0).flatten().astype(np.float32)
                await _process_audio_data(full_audio)
            else:
                enroll_status.value = "No audio recorded."
                
        except Exception as ex:
            enroll_status.value = f"Error: {ex}"
            print(f"Enrollment error: {ex}")
        finally:
            cap.stop()
            btn_start_enroll.disabled = False
            btn_process_file.disabled = False
            page.update()

    # FILE HANDLER
    async def run_file_enrollment():
        path = txt_file_path.value
        if not path or not os.path.exists(path):
            enroll_status.value = "Error: File not found."
            page.update()
            return

        btn_start_enroll.disabled = True
        btn_process_file.disabled = True
        enroll_status.value = "Loading and converting file..."
        enroll_progress.value = None
        page.update()

        try:
            ffmpeg_path = config.get('storage', {}).get('ffmpeg_path', 'ffmpeg')
            # Run in thread to allow UI updates
            audio_data = await asyncio.to_thread(_load_audio_file, path, ffmpeg_path)
            
            if len(audio_data) / 16000 < enroll_duration:
                enroll_status.value = f"Error: Audio too short. Need {enroll_duration}s, got {len(audio_data)/16000:.1f}s."
                enroll_progress.value = 0
            else:
                enroll_status.value = "Processing voice print..."
                await _process_audio_data(audio_data)

        except Exception as e:
            enroll_status.value = f"Error: {e}"
        finally:
            btn_start_enroll.disabled = False
            btn_process_file.disabled = False
            page.update()

    # SHARED PROCESSING
    async def _process_audio_data(audio_data):
        def process():
            success, msg = speaker_verifier.enroll_voice(audio_data, 16000)
            if success:
                speaker_verifier.save_enrollment(enrollment_file)
            return success, msg
        
        try:
            success, msg = await asyncio.to_thread(process)
            if success:
                enroll_status.value = f"Enrollment Complete! {msg}"
                enroll_progress.value = 1.0
            else:
                enroll_status.value = f"Error: {msg}"
                enroll_progress.value = 0
        except Exception as e:
             enroll_status.value = f"Processing error: {e}"

    # Click Listeners
    btn_start_enroll.on_click = lambda e: page.run_task(run_mic_enrollment)
    btn_process_file.on_click = lambda e: page.run_task(run_file_enrollment)
    btn_close = ft.TextButton("Close", on_click=close_dialog)

    # Tabs & Content Area
    content_area = ft.Container(padding=10)
    
    # Define contents
    content_mic = ft.Container(
        content=ft.Column([
            ft.Text("Please read the following text naturally:"),
            ft.Container(content=txt_passage, padding=10, bgcolor=ft.Colors.GREY_900, border_radius=5),
            ft.Divider(),
            ft.Text("Voice Activity:"),
            viz_bars,
            viz_text,
            ft.Container(height=10),
            btn_start_enroll
        ], spacing=5, scroll="auto"),
        padding=10
    )
    
    content_file = ft.Container(
        content=ft.Column([
            ft.Text("Select an audio file containing your voice (mp3, wav, m4a, etc)."),
            ft.Text(f"Must be at least {enroll_duration} seconds long.", size=12, color="grey"),
            ft.Container(height=10),
            txt_file_path,
            ft.Container(height=10),
            btn_process_file
        ], spacing=5, scroll="auto"),
        padding=10
    )

    # Initial content
    content_area.content = content_mic

    # --- Manual Tabs Implementation ---
    # Since ft.Tabs is behaving erratically in this environment, we build a manual tab bar.
    
    btn_tab_mic = ft.TextButton("Microphone", style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.PRIMARY))
    btn_tab_file = ft.TextButton("File Upload", style=ft.ButtonStyle(color=ft.Colors.GREY_400))
    
    def update_tab_styles(idx):
        if idx == 0:
            btn_tab_mic.style.color = ft.Colors.WHITE
            btn_tab_mic.style.bgcolor = ft.Colors.PRIMARY
            btn_tab_file.style.color = ft.Colors.GREY_400
            btn_tab_file.style.bgcolor = None
        else:
            btn_tab_mic.style.color = ft.Colors.GREY_400
            btn_tab_mic.style.bgcolor = None
            btn_tab_file.style.color = ft.Colors.WHITE
            btn_tab_file.style.bgcolor = ft.Colors.PRIMARY
        btn_tab_mic.update()
        btn_tab_file.update()

    def on_tab_mic_click(e):
        content_area.content = content_mic
        content_area.update()
        update_tab_styles(0)

    def on_tab_file_click(e):
        content_area.content = content_file
        content_area.update()
        update_tab_styles(1)

    btn_tab_mic.on_click = on_tab_mic_click
    btn_tab_file.on_click = on_tab_file_click

    tabs_row = ft.Row([btn_tab_mic, btn_tab_file], alignment=ft.MainAxisAlignment.CENTER)

    # Status Info
    status_info = speaker_verifier.get_enrollment_status()
    status_prefix = f"Current: {status_info['embedding_count']} sample(s)." if status_info['enrolled'] else "Not enrolled."

    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Voice Enrollment"),
        content=ft.Container(
            width=500,
            height=600,
            content=ft.Column([
                ft.Text(f"{status_prefix} To use 'Only listen to my voice', we need to learn your filtered voice."),
                ft.Divider(),
                tabs_row, # The manual tabs
                ft.Container(height=10),
                content_area, # The switched content
                ft.Divider(),
                enroll_status,
                enroll_progress
            ], scroll=ft.ScrollMode.AUTO)
        ),
        actions=[btn_close],
    )
    
    dialog_ref[0] = dialog
    return dialog


def create_add_samples_dialog(page: ft.Page, config: dict, speaker_verifier, enrollment_file: str) -> ft.AlertDialog:
    """
    Create Add Samples Dialog (Mic + File).
    """
    # ... Simplified version of enrollment for adding samples ...
    # We can reuse similar logic but simpler text
    
    status_info = speaker_verifier.get_enrollment_status()
    if not status_info['enrolled']:
         # Should handle this check before calling, but safe to return error dialog
         from .dialogs import create_error_dialog
         return create_error_dialog(page, "Error", "Please enroll your voice first.")

    add_duration = 20
    enroll_progress = ft.ProgressBar(width=400, value=0)
    enroll_status = ft.Text(f"Current: {status_info['embedding_count']} samples.")
    
    # Flags
    cancelled = [False]
    dlg_ref = [None]
    
    def close_dlg(e):
        cancelled[0] = True
        if dlg_ref[0]:
            dlg_ref[0].open = False
        page.update()

    # --- SHARED WORKER ---
    async def _process_add(audio_data):
        def process():
            return speaker_verifier.add_voice_sample(audio_data, 16000)
        
        try:
            success, msg, added = await asyncio.to_thread(process)
            if success:
                speaker_verifier.save_enrollment(enrollment_file)
                new_count = speaker_verifier.get_enrollment_status()['embedding_count']
                enroll_status.value = f"Success! {msg}. Total: {new_count}."
            else:
                enroll_status.value = f"Error: {msg}"
            enroll_progress.value = 1.0
        except Exception as e:
            enroll_status.value = f"Error: {e}"

    # --- MIC ---
    viz_bars = ft.ProgressBar(value=0, color=ft.Colors.GREEN, width=400)
    viz_text = ft.Text("Volume: 0%", size=12)
    btn_mic_start = ft.FilledButton("Start Recording")
    
    async def run_mic():
        btn_mic_start.disabled = True
        enroll_status.value = "Recording..."
        page.update()
        
        cap = AudioCapture()
        cap.start()
        chunks = []
        start = time.time()
        
        try:
            while True:
                if cancelled[0] or (time.time() - start >= add_duration):
                    break
                try:
                    chunk = cap.q.get_nowait()
                    chunks.append(chunk)
                    rms = np.sqrt(np.mean(chunk**2))
                    viz_bars.value = min(rms*10, 1.0)
                    viz_text.value = f"{int(viz_bars.value*100)}%"
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                enroll_progress.value = (time.time() - start) / add_duration
                page.update()
                await asyncio.sleep(0.01)
                
            cap.stop()
            viz_bars.value = 0
            if cancelled[0]: return
            
            enroll_status.value = "Processing..."
            enroll_progress.value = None
            page.update()
            
            if chunks:
                full = np.concatenate(chunks).flatten().astype(np.float32)
                await _process_add(full)
            else:
                enroll_status.value = "No audio."
                
        finally:
            cap.stop()
            btn_mic_start.disabled = False
            page.update()

    btn_mic_start.on_click = lambda e: page.run_task(run_mic)

    # --- FILE ---
    txt_file = ft.TextField(label="File Path", text_size=12, expand=True)
    btn_file = ft.FilledButton("Process File", icon=ft.Icons.UPLOAD)
    
    async def run_file():
        path = txt_file.value
        if not path or not os.path.exists(path):
            enroll_status.value = "File not found."
            page.update()
            return
            
        btn_file.disabled = True
        enroll_status.value = "Loading..."
        enroll_progress.value = None
        page.update()
        
        try:
            ffmpeg = config.get('storage', {}).get('ffmpeg_path', 'ffmpeg')
            audio = await asyncio.to_thread(_load_audio_file, path, ffmpeg)
            enroll_status.value = "Processing..."
            await _process_add(audio)
        except Exception as e:
            enroll_status.value = f"Error: {e}"
        finally:
            btn_file.disabled = False
            page.update()
            
    btn_file.on_click = lambda e: page.run_task(run_file)

    # --- TABS ---
    # --- TABS Refactor ---
    content_area = ft.Container(padding=0, expand=True)

    content_mic = ft.Column([
        ft.Text(f"Read anything for {add_duration}s."),
        viz_bars, viz_text, btn_mic_start
    ], spacing=10, scroll="auto")
    
    content_file = ft.Column([
        ft.Text("Select a file with your voice."),
        txt_file, btn_file
    ], spacing=10)

    content_area.content = content_mic

    # --- Manual Tabs Implementation ---
    btn_tab_mic = ft.TextButton("Microphone", style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.PRIMARY))
    btn_tab_file = ft.TextButton("File Upload", style=ft.ButtonStyle(color=ft.Colors.GREY_400))
    
    def update_tab_styles(idx):
        if idx == 0:
            btn_tab_mic.style.color = ft.Colors.WHITE
            btn_tab_mic.style.bgcolor = ft.Colors.PRIMARY
            btn_tab_file.style.color = ft.Colors.GREY_400
            btn_tab_file.style.bgcolor = None
        else:
            btn_tab_mic.style.color = ft.Colors.GREY_400
            btn_tab_mic.style.bgcolor = None
            btn_tab_file.style.color = ft.Colors.WHITE
            btn_tab_file.style.bgcolor = ft.Colors.PRIMARY
        btn_tab_mic.update()
        btn_tab_file.update()

    def on_tab_mic_click(e):
        content_area.content = content_mic
        content_area.update()
        update_tab_styles(0)

    def on_tab_file_click(e):
        content_area.content = content_file
        content_area.update()
        update_tab_styles(1)

    btn_tab_mic.on_click = on_tab_mic_click
    btn_tab_file.on_click = on_tab_file_click

    tabs_row = ft.Row([btn_tab_mic, btn_tab_file], alignment=ft.MainAxisAlignment.CENTER)

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Add Voice Samples"),
        content=ft.Container(
            width=500, height=450,
            content=ft.Column([
                enroll_status,
                enroll_progress,
                ft.Divider(),
                tabs_row,
                ft.Container(height=10),
                content_area
            ])
        ),
        actions=[ft.TextButton("Close", on_click=close_dlg)]
    )
    dlg_ref[0] = dlg
    return dlg
