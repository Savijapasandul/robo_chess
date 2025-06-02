import tkinter as tk
import chess

SQUARE_SIZE = 60
BOARD_SIZE = SQUARE_SIZE * 8
MARGIN = 20  # space for ranks/files
PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}


class ChessGUI:
    def __init__(self, root):
        self.board = chess.Board()
        self.root = root

        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        canvas_width = BOARD_SIZE + MARGIN
        canvas_height = BOARD_SIZE + MARGIN
        self.canvas = tk.Canvas(self.main_frame, width=canvas_width, height=canvas_height)
        self.canvas.pack(side=tk.LEFT)

        self.selected = None
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                x1 = MARGIN + col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

                # Draw piece
                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    self.canvas.create_text(
                        x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2,
                        text=PIECE_UNICODE[piece.symbol()],
                        font=("Arial", 32)
                    )

        # Draw ranks (1–8)
        for row in range(8):
            rank = str(8 - row)
            y = row * SQUARE_SIZE + SQUARE_SIZE // 2
            self.canvas.create_text(MARGIN // 2, y, text=rank, font=("Arial", 14))

        # Draw files (A–H)
        for col in range(8):
            file = chr(ord('A') + col)
            x = MARGIN + col * SQUARE_SIZE + SQUARE_SIZE // 2
            self.canvas.create_text(x, BOARD_SIZE + MARGIN // 2, text=file, font=("Arial", 14))

    def on_click(self, event):
        col = (event.x - MARGIN) // SQUARE_SIZE
        row = 7 - (event.y // SQUARE_SIZE)

        if 0 <= col < 8 and 0 <= row < 8:
            square = chess.square(col, row)
            if self.selected is None:
                if self.board.piece_at(square) and self.board.piece_at(square).color == self.board.turn:
                    self.selected = square
            else:
                move = chess.Move(self.selected, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.post_move_hook()
                self.selected = None
                self.draw_board()

    def post_move_hook(self):
        pass


class ChessGUIWithFeedback(ChessGUI):
    def __init__(self, root):
        super().__init__(root)

        # Right-side panel
        self.info_frame = tk.Frame(self.main_frame, padx=10, pady=10)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.turn_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
        self.turn_label.pack(pady=(0, 10))

        self.move_listbox = tk.Listbox(self.info_frame, width=25, font=("Arial", 12), height=25)
        self.move_listbox.pack(fill=tk.BOTH, expand=True)

        self.update_feedback()

    def update_feedback(self):
        turn = "White" if self.board.turn else "Black"
        self.turn_label.config(text=f"Turn: {turn}")
        self.move_listbox.delete(0, tk.END)

        temp_board = chess.Board()
        moves = list(self.board.move_stack)
        for i in range(0, len(moves), 2):
            move_text = f"{i//2 + 1}. {temp_board.san(moves[i])}"
            temp_board.push(moves[i])
            if i + 1 < len(moves):
                move_text += f" {temp_board.san(moves[i+1])}"
                temp_board.push(moves[i+1])
            self.move_listbox.insert(tk.END, move_text)

    def post_move_hook(self):
        self.update_feedback()


# Main application
root = tk.Tk()
root.title("Chess Thinker GUI")
gui = ChessGUIWithFeedback(root)
root.mainloop()
