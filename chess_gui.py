import tkinter as tk
from tkinter import messagebox
import chess
from stockfish import Stockfish


class RoboChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RoboChess")
        self.board = chess.Board()
        self.selected_piece = None
        self.move_history = []

        # Path to Stockfish binary
        self.stockfish = Stockfish(path="/home/savija/projects/robo_chess/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2", parameters={"Threads": 2, "Minimum Thinking Time": 30})
        self.stockfish.set_skill_level(1)

        self.start_game()

    def start_game(self):
        self.clear_screen()
        self.board.reset()
        self.move_history.clear()
        self.stockfish.set_fen_position(self.board.fen())
        self.create_game_ui()

    def create_game_ui(self):
        container = tk.Frame(self.root, bg="#222")
        container.pack(padx=10, pady=10)

        # Chessboard frame
        board_container = tk.Frame(container, bg="#222")
        board_container.grid(row=0, column=0, padx=10)

        # Labels for files A–H (top and bottom)
        for i in range(8):
            tk.Label(board_container, text=chr(65 + i), fg="white", bg="#222", font=("Arial", 10)).grid(row=0, column=i+1)
            tk.Label(board_container, text=chr(65 + i), fg="white", bg="#222", font=("Arial", 10)).grid(row=9, column=i+1)

        # Labels for ranks 8–1 (left and right)
        for i in range(8):
            tk.Label(board_container, text=str(8 - i), fg="white", bg="#222", font=("Arial", 10)).grid(row=i+1, column=0)
            tk.Label(board_container, text=str(8 - i), fg="white", bg="#222", font=("Arial", 10)).grid(row=i+1, column=9)

        self.canvas = tk.Canvas(board_container, width=480, height=480)
        self.canvas.grid(row=1, column=1, rowspan=8, columnspan=8)
        self.canvas.bind("<Button-1>", self.on_click)

        self.squares = {}
        for r in range(8):
            for c in range(8):
                color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                self.squares[(r, c)] = self.canvas.create_rectangle(c * 60, r * 60, (c + 1) * 60, (r + 1) * 60, fill=color)

        # Side Panel
        side_panel = tk.Frame(container, bg="#333")
        side_panel.grid(row=0, column=1, padx=20)

        self.turn_label = tk.Label(side_panel, text="White's Turn", font=("Arial", 14), fg="white", bg="#333")
        self.turn_label.pack(pady=(0, 10))

        tk.Label(side_panel, text="Move History", font=("Arial", 12), fg="white", bg="#333").pack()
        self.history_box = tk.Text(side_panel, height=20, width=25, font=("Courier", 10), bg="#222", fg="white", state="disabled")
        self.history_box.pack(pady=5)

        reset_button = tk.Button(side_panel, text="Reset Game", command=self.reset_game, font=("Arial", 12), bg="#FF5722", fg="white")
        reset_button.pack(pady=5)

        self.update_board_display()

    def on_click(self, event):
        col = event.x // 60
        row = event.y // 60

        if self.selected_piece:
            start_row, start_col = self.selected_piece[1], self.selected_piece[2]
            start_sq = chess.square(start_col, 7 - start_row)
            end_sq = chess.square(col, 7 - row)
            move = chess.Move(start_sq, end_sq)

            if self.board.piece_at(start_sq).piece_type == chess.PAWN and (7 - row == 0 or 7 - row == 7):
                move.promotion = chess.QUEEN

            if move in self.board.legal_moves:
                san_move = self.board.san(move)  # ✅ Get SAN before pushing
                self.board.push(move)
                self.stockfish.set_fen_position(self.board.fen())
                self.selected_piece = None
                self.log_move(san_move)
                self.update_board_display()

                if self.board.is_game_over():
                    self.game_over_message()
                    return

                self.make_ai_move()
            else:
                messagebox.showerror("Invalid Move", "That move is not allowed.")
        else:
            piece = self.board.piece_at(chess.square(col, 7 - row))
            if piece and piece.color == chess.WHITE:
                self.selected_piece = (piece, row, col)
                self.update_board_display()

    def make_ai_move(self):
        if self.board.is_game_over():
            return

        self.stockfish.set_fen_position(self.board.fen())
        best_move = self.stockfish.get_best_move()
        if best_move:
            move = chess.Move.from_uci(best_move)
            if move in self.board.legal_moves:
                san_move = self.board.san(move)  # ✅ Get SAN before pushing
                self.board.push(move)
                self.log_move(san_move)
                self.update_board_display()

                if self.board.is_game_over():
                    self.game_over_message()

    def update_board_display(self):
        self.canvas.delete("piece")
        for r in range(8):
            for c in range(8):
                square = chess.square(c, 7 - r)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = self.get_piece_symbol(piece)
                    self.draw_piece(c * 60 + 30, r * 60 + 30, symbol)

        turn = "White" if self.board.turn else "Black"
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
            self.start_game()

    def game_over_message(self):
        if self.board.is_checkmate():
            winner = "Black Wins" if self.board.turn else "White Wins"
        elif self.board.is_stalemate():
            winner = "Draw (Stalemate)"
        elif self.board.is_insufficient_material():
            winner = "Draw (Insufficient Material)"
        else:
            winner = "Game Over"
        messagebox.showinfo("Game Over", winner)
        self.start_game()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RoboChessApp(root)
    root.mainloop()
