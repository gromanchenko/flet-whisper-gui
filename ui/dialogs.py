"""
Dialog factories for Transcriber UI.
These are factory functions that create Flet dialogs.
"""
import flet as ft


def create_confirm_dialog(page: ft.Page, title: str, content: str, on_confirm) -> ft.AlertDialog:
    """
    Create a confirmation dialog with Cancel/Confirm actions.
    
    Args:
        page: Flet page for dialog updates
        title: Dialog title
        content: Dialog message
        on_confirm: Callback when user confirms
        
    Returns:
        AlertDialog instance (caller must set .open = True and page.update())
    """
    # Create dialog first so closures can reference it
    dialog = ft.AlertDialog(
        title=ft.Text(title),
        content=ft.Text(content),
        actions_alignment=ft.MainAxisAlignment.END,
    )
    
    def close_dlg(e):
        dialog.open = False
        page.update()
        
    def confirm_action(e):
        dialog.open = False
        page.update()
        on_confirm(e)
    
    dialog.actions = [
        ft.TextButton("Cancel", on_click=close_dlg),
        ft.TextButton("Delete", on_click=confirm_action, style=ft.ButtonStyle(color=ft.Colors.RED)),
    ]
    
    return dialog


def create_error_dialog(page: ft.Page, title: str, message: str) -> ft.AlertDialog:
    """
    Create an error dialog with a single OK button.
    
    Args:
        page: Flet page for dialog updates
        title: Dialog title
        message: Error message
        
    Returns:
        AlertDialog instance
    """
    # Create dialog first so closure can reference it
    dialog = ft.AlertDialog(
        title=ft.Text(title),
        content=ft.Text(message),
    )
    
    def close_dlg(e):
        dialog.open = False
        page.update()
    
    dialog.actions = [ft.TextButton("OK", on_click=close_dlg)]
    
    return dialog


def create_diagnostic_dialog(page: ft.Page, config: dict) -> ft.AlertDialog:
    """
    Create a diagnostic dialog showing effective configuration.
    
    Args:
        page: Flet page for dialog updates
        config: Configuration dict to display
        
    Returns:
        AlertDialog instance
    """
    # Format config as readable text
    lines = ["# Effective Configuration\n"]
    
    for section in ['whisper', 'vad', 'storage', 'speaker']:
        section_data = config.get(section, {})
        lines.append(f"\n[{section}]")
        for key, value in section_data.items():
            lines.append(f"  {key} = {repr(value)}")
    
    config_text = "\n".join(lines)
    
    # Create dialog first so closure can reference it
    dialog = ft.AlertDialog(
        title=ft.Text("Diagnostic: Effective Configuration"),
        content=ft.Container(
            width=500,
            height=400,
            content=ft.Column([
                ft.Text("Current runtime configuration values:", size=12, italic=True),
                ft.Container(
                    border=ft.Border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                    padding=10,
                    expand=True,
                    content=ft.ListView(
                        expand=True,
                        controls=[ft.Text(config_text, font_family="monospace", size=12)]
                    )
                )
            ], expand=True)
        ),
    )
    
    def close_dlg(e):
        dialog.open = False
        page.update()
    
    dialog.actions = [ft.TextButton("Close", on_click=close_dlg)]
    
    return dialog
