import flet as ft
import chess
from chess_logic import ChessLogic

class ChessGUI:
    def __init__(self, page: ft.Page, device_type: str, logic: ChessLogic):
        self.page = page
        self.device_type = device_type
        self.logic = logic
        self.selected_square = None
        self.legal_moves = []
        self.piece_symbols = self.load_piece_symbols()
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
            value=f"Turn: {'White' if self.logic.board.turn == chess.WHITE else 'Black'}",
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
        self.reset_button = ft.ElevatedButton(text="Reset", on_click=self.on_reset)
        self.settings_button = ft.IconButton(
            icon=ft.Icons.SETTINGS,
            on_click=self.show_settings_dialog
        )
        
        self.controls_row = ft.Row(
            [self.settings_button, self.reset_button],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
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
        self.update_board()

    def create_board_tiles(self):
        # Create board from black's perspective (rotated)
        for rank in range(7, -1, -1):
            for file in range(8):
                square = rank * 8 + file
                tile_color = "#F0D9B5" if (rank + file) % 2 == 0 else "#B58863"
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
        board = self.logic.get_board_state()
        for i, tile in enumerate(self.board_grid.controls[:]):
            # Convert display index to chess square
            display_index = i
            rank = 7 - (display_index // 8)
            file = display_index % 8
            square = rank * 8 + file
            
            piece = board.piece_at(square)
            base_color = "#F0D9B5" if (rank + file) % 2 == 0 else "#B58863"
            
            # Highlighting
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
        board = self.logic.get_board_state()
        self.turn_label.value = f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}"
        if board.is_check():
            self.turn_label.value += " (CHECK)"
        self.turn_label.update()

    def on_square_click(self, e):
        square = e.control.data
        
        # Convert display index to chess square
        display_index = self.board_grid.controls.index(e.control)
        rank = 7 - (display_index // 8)
        file = display_index % 8
        square = rank * 8 + file

        if self.selected_square is None:
            # Select a piece (only allow white to move)
            piece = self.logic.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
                self.legal_moves = self.logic.get_legal_moves(square)
        else:
            # Try to move the selected piece
            move = chess.Move(self.selected_square, square)
            
            # Check if promotion is needed
            if self.logic.is_pawn_promotion(move):
                self.promote_pawn(move)
            else:
                if move in self.logic.board.legal_moves:
                    self.logic.make_move(move)
                    self.selected_square = None
                    self.legal_moves = []
                    self.update_board()
                    
                    # AI's turn (black)
                    if not self.logic.board.is_game_over():
                        self.logic.make_ai_move()
                        self.update_board()
                        self.check_game_over()
                else:
                    # Deselect if invalid move
                    self.selected_square = None
                    self.legal_moves = []
        
        self.update_board()
        self.check_game_over()

    def promote_pawn(self, move):
        self.selected_square = None
        self.legal_moves = []
        
        # Create promotion dialog
        self.promotion_dialog = ft.AlertDialog(
            title=ft.Text("Pawn Promotion"),
            content=ft.Text("Choose a piece to promote to:"),
            actions=[
                ft.TextButton("Queen", on_click=lambda e: self.complete_promotion(e, move, chess.QUEEN)),
                ft.TextButton("Rook", on_click=lambda e: self.complete_promotion(e, move, chess.ROOK)),
                ft.TextButton("Bishop", on_click=lambda e: self.complete_promotion(e, move, chess.BISHOP)),
                ft.TextButton("Knight", on_click=lambda e: self.complete_promotion(e, move, chess.KNIGHT)),
            ]
        )
        self.page.dialog = self.promotion_dialog
        self.promotion_dialog.open = True
        self.page.update()

    def complete_promotion(self, e, move, promotion_piece):
        self.promotion_dialog.open = False
        self.logic.make_move(move, promotion_piece)
        self.page.update()

        # Check right away if the *promotion* ended the game
        self.check_game_over()

        # AI's turn (black)
        if not self.logic.board.is_game_over():
            self.logic.make_ai_move()
            self.update_board()
            self.check_game_over()

    def on_reset(self, e):
        self.logic.reset_game()
        self.selected_square = None
        self.legal_moves = []
        self.update_board()

    def show_settings_dialog(self, e):
        print("⚙️ show_settings_dialog() called")
        self.ai_difficulty_slider = ft.Slider(
            min=1, 
            max=3, 
            divisions=2,
            value=self.logic.ai_difficulty,
            label="{value}",
            on_change=self.on_ai_difficulty_change
        )

        self.settings_dialog = ft.AlertDialog(
            title=ft.Text("Game Settings"),
            content=ft.Column([
                ft.Text("AI Difficulty Level"),
                self.ai_difficulty_slider
            ]),
            actions=[
                ft.TextButton("OK", on_click=self.close_settings_dialog),
            ],
        )
        self.page.dialog = self.settings_dialog
        self.settings_dialog.open = True
        self.page.update()

    def close_settings_dialog(self, e):
        self.settings_dialog.open = False
        self.page.update()

    def on_ai_difficulty_change(self, e):
        self.logic.set_difficulty(int(e.control.value))

    def check_game_over(self):
        result = self.logic.check_game_over()
        if result:
            self.show_game_over_dialog(result)

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