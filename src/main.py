import flet as ft
import chess
import chess.engine
import random

class ChessApp:
    def __init__(self, page: ft.Page, device_type: str):
        self.page = page
        self.device_type = device_type
        self.board = chess.Board()
        self.move_history = []
        self.selected_square = None
        self.legal_moves = []
        self.piece_symbols = self.load_piece_symbols()
        self.ai_difficulty = 1
        self.game_started = False
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"/home/savija/projects/robo_chess/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
        )
        self.setup_ui()

    def load_piece_symbols(self):
        return {
            "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "P": "♙",
            "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", "p": "♟"
        }

    def setup_ui(self):
        self.page.title = "Chess App"
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.padding = 20

        self.turn_label = ft.Text(
            value=f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}",
            size=20,
            weight=ft.FontWeight.BOLD,
        )

        # Create board grid
        self.board_grid = ft.GridView(
            expand=True,
            runs_count=8,
            child_aspect_ratio=1,
            spacing=0,
            run_spacing=0,
        )
        self.create_board_tiles()
        
        # Create controls
        self.undo_button = ft.ElevatedButton(text="Undo", on_click=self.on_undo, disabled=True)
        self.redo_button = ft.ElevatedButton(text="Redo", on_click=self.on_redo, disabled=True)
        self.reset_button = ft.ElevatedButton(text="Reset", on_click=self.on_reset)
        self.start_button = ft.ElevatedButton(text="Start", on_click=self.on_start)
        
        self.controls_row = ft.Row(
            [self.undo_button, self.redo_button, self.reset_button, self.start_button],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        )

        self.page.add(
            ft.Column(
                [
                    ft.Row([self.turn_label], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Container(
                        content=self.board_grid,
                        alignment=ft.alignment.center,
                        width=self.calculate_board_size(),
                        height=self.calculate_board_size(),
                        border=ft.border.all(2, ft.Colors.BLACK),),
                    self.controls_row,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )

        self.show_initial_dialog()
        self.update_board()

    def create_board_tiles(self):
        for square in range(64):
            row, col = divmod(square, 8)
            tile_color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            tile = ft.Container(
                bgcolor=tile_color,
                alignment=ft.alignment.center,
                data=square,
                on_click=self.on_square_click,
                content=ft.Text("", size=24),
            )
            self.board_grid.controls.append(tile)

    def calculate_board_size(self):
        screen_width = self.page.width or 400
        screen_height = self.page.height or 600
        if self.device_type == "Phone":
            return min(screen_width, screen_height) * 0.9
        elif self.device_type == "Tablet":
            return min(screen_width, screen_height) * 0.7
        else:
            return min(screen_width, screen_height) * 0.6

    def update_board(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            tile = self.board_grid.controls[square]
            
            # Reset tile appearance
            row, col = divmod(square, 8)
            base_color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            
            if square == self.selected_square:
                tile.bgcolor = "#6B8E23"  # Highlight selected square
            elif square in self.legal_moves:
                tile.bgcolor = "#ADD8E6"  # Highlight legal moves
            else:
                tile.bgcolor = base_color
            
            # Update piece display
            if piece:
                symbol = self.piece_symbols[piece.symbol()]
                tile.content = ft.Text(
                    value=symbol,
                    size=24,
                    color=ft.Colors.BLACK if piece.color == chess.BLACK else ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD
                )
            else:
                tile.content = ft.Text("", size=24)
                
            tile.update()

        self.update_turn_label()

    def update_turn_label(self):
        self.turn_label.value = f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}"
        if self.board.is_check():
            self.turn_label.value += " (CHECK)"
        self.turn_label.update()

    def on_square_click(self, e):
        if not self.game_started:
            return

        square = e.control.data
        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:  # Only allow white to move
                self.selected_square = square
                self.legal_moves = [
                    move.to_square 
                    for move in self.board.legal_moves 
                    if move.from_square == square
                ]
        else:
            # Try to move the selected piece
            move = chess.Move(self.selected_square, square)
            
            # Check if promotion is needed
            if move.to_square in chess.SquareSet(chess.BB_BACKRANKS) and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                self.promote_pawn(move)
            else:
                if move in self.board.legal_moves:
                    self.make_move(move)
                else:
                    # Deselect if invalid move
                    self.selected_square = None
                    self.legal_moves = []
        
        self.update_board()

    def promote_pawn(self, move):
        self.selected_square = None
        self.legal_moves = []
        
        # Create promotion dialog
        self.promotion_dialog = ft.AlertDialog(
            title=ft.Text("Pawn Promotion"),
            content=ft.Text("Choose a piece to promote to:"),
            actions=[
                ft.TextButton("Queen", on_click=lambda e: self.complete_move(move, chess.QUEEN)),
                ft.TextButton("Rook", on_click=lambda e: self.complete_move(move, chess.ROOK)),
                ft.TextButton("Bishop", on_click=lambda e: self.complete_move(move, chess.BISHOP)),
                ft.TextButton("Knight", on_click=lambda e: self.complete_move(move, chess.KNIGHT)),
            ]
        )
        self.page.dialog = self.promotion_dialog
        self.promotion_dialog.open = True
        self.page.update()

    def complete_move(self, move, promotion_piece):
        self.promotion_dialog.open = False
        move.promotion = promotion_piece
        self.make_move(move)
        self.page.update()

    def make_move(self, move):
        self.board.push(move)
        self.move_history.append(move)
        self.selected_square = None
        self.legal_moves = []
        self.undo_button.disabled = False
        self.check_game_over()
        
        # AI's turn (black)
        if not self.board.is_game_over() and self.board.turn == chess.BLACK:
            self.make_ai_move()

    def on_undo(self, e):
        if len(self.move_history) > 0:
            self.board.pop()
            self.move_history.pop()
            self.selected_square = None
            self.legal_moves = []
            self.undo_button.disabled = len(self.move_history) == 0
            self.update_board()

    def on_redo(self, e):
        # Redo functionality not implemented
        pass

    def on_reset(self, e):
        self.board.reset()
        self.move_history = []
        self.selected_square = None
        self.legal_moves = []
        self.undo_button.disabled = True
        self.game_started = False
        self.update_board()
        self.show_initial_dialog()

    def on_start(self, e):
        self.game_started = True
        if self.dialog_container in self.page.overlay:
            self.page.overlay.remove(self.dialog_container)
        self.page.update()
        # AI moves first if it's black's turn
        if self.board.turn == chess.BLACK:
            self.make_ai_move()

    def show_initial_dialog(self):
        self.ai_difficulty_slider = ft.Slider(
            min=1, 
            max=3, 
            divisions=2,
            value=self.ai_difficulty,
            label="{value}",
            on_change=self.on_ai_difficulty_change
        )
        self.start_button = ft.ElevatedButton(text="Start Game", on_click=self.on_start)

        self.dialog_container = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Game Settings", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text("AI Difficulty Level"),
                    self.ai_difficulty_slider,
                    ft.Row([self.start_button], alignment=ft.MainAxisAlignment.CENTER),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            ),
            alignment=ft.alignment.center,
            bgcolor=ft.Colors.WHITE,
            padding=30,
            border_radius=10,
            shadow=ft.BoxShadow(blur_radius=20, spread_radius=5, color=ft.Colors.BLACK12),
            width=300,
        )

        self.page.overlay.append(self.dialog_container)
        self.page.update()

    def on_ai_difficulty_change(self, e):
        self.ai_difficulty = int(e.control.value)

    def make_ai_move(self):
        # Set thinking time based on difficulty
        think_time = 0.5 * self.ai_difficulty
        
        try:
            result = self.engine.play(
                self.board, 
                chess.engine.Limit(time=think_time),
                options={"Skill Level": self.ai_difficulty * 5}
            )
            self.make_move(result.move)
        except Exception as e:
            print(f"AI move error: {e}")
            # Fallback to random move if engine fails
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                self.make_move(random.choice(legal_moves))

    def check_game_over(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            self.show_game_over_dialog(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.show_game_over_dialog("Stalemate! Game is a draw.")
        elif self.board.is_insufficient_material():
            self.show_game_over_dialog("Draw by insufficient material!")
        elif self.board.is_seventyfive_moves():
            self.show_game_over_dialog("Draw by 75-move rule!")
        elif self.board.is_fivefold_repetition():
            self.show_game_over_dialog("Draw by repetition!")

    def show_game_over_dialog(self, message):
        dialog = ft.AlertDialog(
            title=ft.Text("Game Over"),
            content=ft.Text(message),
            actions=[
                ft.TextButton("New Game", on_click=self.on_reset),
                ft.TextButton("Close", on_click=lambda e: self.close_dialog(e)),
            ],
        )
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def close_dialog(self, e):
        self.page.dialog.open = False
        self.page.update()

    def __del__(self):
        try:
            self.engine.quit()
        except Exception:
            pass

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

    ChessApp(page, device_type)

ft.app(target=main)
