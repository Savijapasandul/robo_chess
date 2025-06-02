import tkinter as tk
import chess
from tkinter import messagebox

SQUARE_SIZE = 60
ANIMATION_STEPS = 15
ANIMATION_DELAY = 0.03  # seconds per animation step

PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

class ChessGUIWithFeedback:
    def __init__(self, root):
        self.board = chess.Board()
        self.root = root
        self.selected = None
        self.animating = False

        # Main frame to hold board and feedback side by side
        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        # Canvas for chessboard + rails, add margin for rails (20px)
        canvas_width = 8 * SQUARE_SIZE + 40
        canvas_height = 8 * SQUARE_SIZE + 40
        self.canvas = tk.Canvas(self.main_frame, width=canvas_width, height=canvas_height)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Info frame on the right for feedback
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.grid(row=0, column=1, sticky='ns', padx=10, pady=10)

        self.turn_label = tk.Label(self.info_frame, text="", font=("Arial", 16, "bold"), pady=10)
        self.turn_label.pack()

        self.move_listbox = tk.Listbox(self.info_frame, width=25, height=20, font=("Consolas", 12))
        self.move_listbox.pack(fill=tk.BOTH, expand=True)

        self.new_game_button = tk.Button(self.info_frame, text="New Game", command=self.new_game, state=tk.DISABLED)
        self.new_game_button.pack(pady=10)

        self.draw_board()
        self.update_feedback()

        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        self.canvas.delete("all")
        # Draw reference rails (letters and numbers)
        for i in range(8):
            # Letters at bottom and top
            letter = chr(ord('A') + i)
            x = 20 + i * SQUARE_SIZE + SQUARE_SIZE // 2
            self.canvas.create_text(x, 8 * SQUARE_SIZE + 30, text=letter, font=("Arial", 14, "bold"))
            self.canvas.create_text(x, 10, text=letter, font=("Arial", 14, "bold"))
            # Numbers at left and right
            number = str(8 - i)
            y = 20 + i * SQUARE_SIZE + SQUARE_SIZE // 2
            self.canvas.create_text(10, y, text=number, font=("Arial", 14, "bold"))
            self.canvas.create_text(8 * SQUARE_SIZE + 30, y, text=number, font=("Arial", 14, "bold"))

        # Draw squares and pieces
        for row in range(8):
            for col in range(8):
                x1 = 20 + col * SQUARE_SIZE
                y1 = 20 + row * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE

                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    tag = f"piece_{col}_{row}"
                    self.canvas.create_text(
                        x1 + SQUARE_SIZE // 2,
                        y1 + SQUARE_SIZE // 2,
                        text=PIECE_UNICODE[piece.symbol()],
                        font=("Arial", 32),
                        tags=tag
                    )

        # Highlight selected square
        if self.selected is not None:
            sel_col = chess.square_file(self.selected)
            sel_row = 7 - chess.square_rank(self.selected)
            x1 = 20 + sel_col * SQUARE_SIZE
            y1 = 20 + sel_row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3)

    def update_feedback(self):
        turn = "White" if self.board.turn else "Black"
        status_msg = f"Turn: {turn}"

        # Add check info if not game over
        if not self.board.is_game_over():
            if self.board.is_check():
                status_msg += " - Check!"
        else:
            # Game is over, determine reason
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                status_msg = f"Checkmate! {winner} wins!"
            elif self.board.is_stalemate():
                status_msg = "Stalemate! Draw!"
            elif self.board.is_insufficient_material():
                status_msg = "Draw due to insufficient material."
            elif self.board.can_claim_fifty_moves():
                status_msg = "Draw by fifty-move rule."
            elif self.board.can_claim_threefold_repetition():
                status_msg = "Draw by threefold repetition."
            else:
                status_msg = "Game over."

        self.turn_label.config(text=status_msg)

        self.move_listbox.delete(0, tk.END)
        moves = list(self.board.move_stack)

        temp_board = chess.Board()
        for i in range(0, len(moves), 2):
            san_1 = temp_board.san(moves[i])
            temp_board.push(moves[i])
            move_text = f"{i // 2 + 1}. {san_1}"

            if i + 1 < len(moves):
                san_2 = temp_board.san(moves[i + 1])
                temp_board.push(moves[i + 1])
                move_text += f" {san_2}"

            self.move_listbox.insert(tk.END, move_text)


    def new_game(self):
        self.board.reset()
        self.selected = None
        self.new_game_button.config(state=tk.DISABLED)
        self.draw_board()
        self.update_feedback()

    def check_game_end(self):
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                msg = "White wins!"
            elif result == "0-1":
                msg = "Black wins!"
            else:
                msg = "Draw!"
            answer = messagebox.askyesno("Game Over", f"{msg}\nStart a new game?")
            if answer:
                self.new_game_button.config(state=tk.NORMAL)
                self.new_game()
            else:
                self.new_game_button.config(state=tk.NORMAL)

    def animate_move(self, move):
        self.animating = True

        start_square = move.from_square
        end_square = move.to_square

        start_col = chess.square_file(start_square)
        start_row = 7 - chess.square_rank(start_square)
        end_col = chess.square_file(end_square)
        end_row = 7 - chess.square_rank(end_square)

        piece = self.board.piece_at(start_square)
        if not piece:
            self.animating = False
            return

        piece_symbol = PIECE_UNICODE[piece.symbol()]

        start_x = 20 + start_col * SQUARE_SIZE + SQUARE_SIZE // 2
        start_y = 20 + start_row * SQUARE_SIZE + SQUARE_SIZE // 2
        end_x = 20 + end_col * SQUARE_SIZE + SQUARE_SIZE // 2
        end_y = 20 + end_row * SQUARE_SIZE + SQUARE_SIZE // 2

        # Remove the piece from start square during animation
        self.canvas.delete(f"piece_{start_col}_{start_row}")

        step = 0
        moving_piece = self.canvas.create_text(start_x, start_y, text=piece_symbol, font=("Arial", 32), tags="anim_piece")

        def step_animation():
            nonlocal step
            if step >= ANIMATION_STEPS:
                self.canvas.delete("anim_piece")
                self.animating = False
                # After animation finishes, push move, redraw and update feedback
                self.board.push(move)
                self.selected = None
                self.draw_board()
                self.update_feedback()
                self.check_game_end()
                return

            interp_x = start_x + (end_x - start_x) * step / ANIMATION_STEPS
            interp_y = start_y + (end_y - start_y) * step / ANIMATION_STEPS
            self.canvas.coords(moving_piece, interp_x, interp_y)
            step += 1
            self.canvas.after(int(ANIMATION_DELAY * 1000), step_animation)

        step_animation()

    def ask_promotion_buttons(self):
        # Popup window to select promotion piece by buttons
        promo_win = tk.Toplevel(self.root)
        promo_win.title("Pawn Promotion")
        promo_win.geometry("+%d+%d" % (self.root.winfo_rootx()+100, self.root.winfo_rooty()+100))
        promo_win.transient(self.root)  # set to be on top of root
        
        promo_win.update_idletasks()
        promo_win.deiconify()
        promo_win.grab_set()  # modal

        chosen = {'piece': None}

        def choose(piece_type):
            chosen['piece'] = piece_type
            promo_win.destroy()

        label = tk.Label(promo_win, text="Promote to:", font=("Arial", 14))
        label.pack(padx=10, pady=10)

        buttons_frame = tk.Frame(promo_win)
        buttons_frame.pack(padx=10, pady=10)

        pieces = [
            ('Queen', chess.QUEEN),
            ('Rook', chess.ROOK),
            ('Bishop', chess.BISHOP),
            ('Knight', chess.KNIGHT),
        ]

        for (name, pt) in pieces:
            btn = tk.Button(buttons_frame, text=name, width=8, command=lambda pt=pt: choose(pt))
            btn.pack(side=tk.LEFT, padx=5)

        promo_win.wait_window()
        return chosen['piece'] if chosen['piece'] is not None else chess.QUEEN


    def on_click(self, event):
        if self.animating:
            return  # ignore clicks while animating

        x_click = event.x
        y_click = event.y

        # Ignore clicks on rails area (less than 20 pixels margin)
        if x_click < 20 or y_click < 20:
            return

        col = (x_click - 20) // SQUARE_SIZE
        row = (y_click - 20) // SQUARE_SIZE

        if not (0 <= col < 8 and 0 <= row < 8):
            return

        square = chess.square(col, 7 - row)

        if self.selected is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected = square
                self.draw_board()
        else:
            piece = self.board.piece_at(self.selected)
            if piece is None:
                self.selected = None
                self.draw_board()
                return

            promotion_rank = 7 if piece.color == chess.WHITE else 0
            if piece.piece_type == chess.PAWN and chess.square_rank(square) == promotion_rank:
                promo_piece = self.ask_promotion_buttons()
                move = chess.Move(self.selected, square, promotion=promo_piece)
            else:
                move = chess.Move(self.selected, square)

            if move in self.board.legal_moves:
                self.animate_move(move)
            else:
                clicked_piece = self.board.piece_at(square)
                if clicked_piece and clicked_piece.color == self.board.turn:
                    self.selected = square
                else:
                    self.selected = None
                self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess Thinker GUI")
    ChessGUIWithFeedback(root)
    root.mainloop()
