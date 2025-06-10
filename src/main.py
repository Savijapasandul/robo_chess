import flet as ft
from chess_logic import ChessLogic
from chess_gui import ChessGUI

def main(page: ft.Page):
    page.title = "Chess App"
    page.window_min_width = 400
    page.window_min_height = 600
    page.update()

    # Detect device type
    user_agent = page.client_user_agent.lower() if page.client_user_agent else ""
    if "mobile" in user_agent:
        device_type = "Phone"
    elif "tablet" in user_agent:
        device_type = "Tablet"
    else:
        device_type = "Desktop"

    # Initialize game logic
    logic = ChessLogic()
    logic.start_engine(r"/home/savija/projects/robo_chess/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2")
    
    # Initialize GUI
    gui = ChessGUI(page, device_type, logic)
    
    # Clean up when app closes
    page.on_close = lambda: logic.close_engine()

ft.app(target=main)