import tkinter as tk
from tkinter import messagebox
import chess
from PIL import Image, ImageTk
import time

SQUARE_SIZE = 60

class ChessGUI:
    def __init__(self, root, logic, engine):
        self.root = root
        self.logic = logic
        self.engine = engine
        self.selected_piece = None
        self.move_history = []

        self.animation_speed = 1.0  # multiplier for animation speed
        self.create_game_ui()
        self.update_board_display()

    def create_game_ui(self):
        container = tk.Frame(self.root, bg="#222")
        container.pack(padx=10, pady=10)

        board_frame = tk.Frame(container, bg="#222")
        board_frame.grid(row=0, column=0, padx=10)

        for i in range(8):
            tk.Label(board_frame, text=chr(65 + i), fg="white", bg="#222", font=("Arial", 10)).grid(row=0, column=i+1)
            tk.Label(board_frame, text=chr(65 + i), fg="white", bg="#222", font=("Arial", 10)).grid(row=9, column=i+1)
        for i in range(8):
            tk.Label(board_frame, text=str(8 - i), fg="white", bg="#222", font=("Arial", 10)).grid(row=i+1, column=0)
            tk.Label(board_frame, text=str(8 - i), fg="white", bg="#222", font=("Arial", 10)).grid(row=i+1, column=9)

        self.canvas = tk.Canvas(board_frame, width=8*SQUARE_SIZE, height=8*SQUARE_SIZE)
        self.canvas.grid(row=1, column=1, rowspan=8, columnspan=8)
        self.canvas.bind("<Button-1>", self.on_click)

        self.squares = {}
        for r in range(8):
            for c in range(8):
                color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                self.squares[(r, c)] = self.canvas.create_rectangle(c*SQUARE_SIZE, r*SQUARE_SIZE, (c+1)*SQUARE_SIZE, (r+1)*SQUARE_SIZE, fill=color)

        side_panel = tk.Frame(container, bg="#333")
        side_panel.grid(row=0, column=1, padx=20)

        self.turn_label = tk.Label(side_panel, text="White's Turn", font=("Arial", 14), fg="white", bg="#333")
        self.turn_label.pack(pady=(0, 10))

        tk.Label(side_panel, text="Move History", font=("Arial", 12), fg="white", bg="#333").pack()
        self.history_box = tk.Text(side_panel, height=20, width=25, font=("Courier", 10), bg="#222", fg="white", state="disabled")
        self.history_box.pack(pady=5)

        reset_button = tk.Button(side_panel, text="Reset Game", command=self.reset_game, font=("Arial", 12), bg="#FF5722", fg="white")
        reset_button.pack(pady=5)

        # Slider for animation speed
        tk.Label(side_panel, text="Animation Speed", font=("Arial", 12), fg="white", bg="#333").pack(pady=(10, 0))
        self.speed_slider = tk.Scale(side_panel, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#333", fg="white",
                                     command=self.on_speed_change)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(pady=5)

        # Slider for difficulty (skill level)
        tk.Label(side_panel, text="Difficulty Level", font=("Arial", 12), fg="white", bg="#333").pack(pady=(10, 0))
        self.diff_slider = tk.Scale(side_panel, from_=1, to=20, orient=tk.HORIZONTAL, bg="#333", fg="white",
                                    command=self.on_difficulty_change)
        self.diff_slider.set(self.engine.skill_level)
        self.diff_slider.pack(pady=5)

    def on_speed_change(self, val):
        self.animation_speed = float(val)

    def on_difficulty_change(self, val):
        skill = int(float(val))
        self.engine.set_skill_level(skill)

    def animate_move(self, start_col, start_row, end_col, end_row, symbol):
        dx = (end_col - start_col) * SQUARE_SIZE
        dy = (end_row - start_row) * SQUARE_SIZE
        steps = 10
        delay = int(20 / self.animation_speed)  # Adjust delay by speed multiplier
        x0 = start_col * SQUARE_SIZE + SQUARE_SIZE // 2
        y0 = start_row * SQUARE_SIZE + SQUARE_SIZE // 2

        temp_piece = self.canvas.create_text(x0, y0, text=symbol, font=("Arial", 32), tags="anim")

        def step(n):
            if n > steps:
                self.canvas.delete("anim")
                self.update_board_display()
                return
            self.canvas.move(temp_piece, dx / steps, dy / steps)
            self.canvas.after(delay, step, n + 1)

        step(1)

    def on_click(self, event):
        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE
        clicked_sq = chess.square(col, 7 - row)
        piece = self.logic.board.piece_at(clicked_sq)

        if self.selected_piece:
            start_sq = chess.square(self.selected_piece[2], 7 - self.selected_piece[1])
            move = chess.Move(start_sq, clicked_sq)

            if piece and piece.color == chess.WHITE:
                self.selected_piece = (piece, row, col)
                self.update_board_display()
                return

            if self.logic.board.piece_at(start_sq).piece_type == chess.PAWN and (7 - row == 0 or 7 - row == 7):
                move.promotion = chess.QUEEN

            if move in self.logic.board.legal_moves:
                symbol = self.get_piece_symbol(self.logic.board.piece_at(start_sq))
                san_move = self.logic.board.san(move)
                self.logic.push_move(move)
                self.animate_move(self.selected_piece[2], self.selected_piece[1], col, row, symbol)
                self.engine.set_fen(self.logic.board.fen())
                self.selected_piece = None
                self.log_move(san_move)

                self.inform_check()  # Inform if player is in check after move

                if self.logic.is_game_over():
                    self.game_over_message()
                    return

                self.make_ai_move()
            else:
                self.selected_piece = None
                self.update_board_display()
        else:
            if piece and piece.color == chess.WHITE:
                self.selected_piece = (piece, row, col)
                self.update_board_display()

    def make_ai_move(self):
        if self.logic.is_game_over():
            return
        self.engine.set_fen(self.logic.board.fen())
        best_move = self.engine.get_best_move()
        if best_move:
            move = chess.Move.from_uci(best_move)
            if move in self.logic.legal_moves():
                san_move = self.logic.board.san(move)
                start_col, start_row = chess.square_file(move.from_square), 7 - chess.square_rank(move.from_square)
                end_col, end_row = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
                symbol = self.get_piece_symbol(self.logic.board.piece_at(move.from_square))
                self.logic.push_move(move)
                self.log_move(san_move)
                self.animate_move(start_col, start_row, end_col, end_row, symbol)

                self.inform_check()  # Inform if player is in check after AI move

                if self.logic.is_game_over():
                    self.game_over_message()

    def inform_check(self):
        if self.logic.board.is_check():
            messagebox.showwarning("Check!", "Your king is in check! Please move out of check.")

    def update_board_display(self):
        self.canvas.delete("piece")
        for r in range(8):
            for c in range(8):
                square = chess.square(c, 7 - r)
                piece = self.logic.board.piece_at(square)
                if piece:
                    symbol = self.get_piece_symbol(piece)
                    self.draw_piece(c*SQUARE_SIZE + SQUARE_SIZE//2, r*SQUARE_SIZE + SQUARE_SIZE//2, symbol)
        turn = "White" if self.logic.get_turn() else "Black"
        self.turn_label.config(text=f"{turn}'s Turn")

    def draw_piece(self, x, y, symbol):
        self.canvas.create_text(x, y, text=symbol, font=("Arial", 32), tags="piece")

    def get_piece_symbol(self, piece):
        symbols = {
            chess.PAWN: '♟' if piece.color == chess.BLACK else '♙',
            chess.ROOK: '♜' if piece.color == chess.BLACK else '♖',
            chess.KNIGHT: '♞' if piece.color == chess.BLACK else '♘',
            chess.BISHOP: '♝' if piece.color == chess.BLACK else '♗',
            chess.QUEEN: '♛' if piece.color == chess.BLACK else '♕',
            chess.KING: '♚' if piece.color == chess.BLACK else '♔',
        }
        return symbols[piece.piece_type]

    def log_move(self, san):
        self.move_history.append(san)
        self.history_box.config(state="normal")
        self.history_box.delete(1.0, tk.END)
        for idx in range(0, len(self.move_history), 2):
            line = f"{idx//2 + 1}. {self.move_history[idx]}"
            if idx + 1 < len(self.move_history):
                line += f"  {self.move_history[idx + 1]}"
            self.history_box.insert(tk.END, line + "\n")
        self.history_box.config(state="disabled")

    def reset_game(self):
        if messagebox.askyesno("Reset Game", "Do you really want to reset the game?"):
            self.logic.reset()
            self.move_history = []
            self.engine.set_fen(self.logic.board.fen())
            self.update_board_display()
            self.log_move("Game Reset")

    def game_over_message(self):
        if self.logic.board.is_checkmate():
            winner = "Black Wins" if self.logic.get_turn() else "White Wins"
        elif self.logic.board.is_stalemate():
            winner = "Draw (Stalemate)"
        elif self.logic.board.is_insufficient_material():
            winner = "Draw (Insufficient Material)"
        else:
            winner = "Game Over"
        messagebox.showinfo("Game Over", winner)
        self.reset_game()
