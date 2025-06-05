import chess

class ChessLogic:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def push_move(self, move):
        self.board.push(move)

    def legal_moves(self):
        return list(self.board.legal_moves)

    def is_game_over(self):
        return self.board.is_game_over()

    def result(self):
        if self.board.is_checkmate():
            return "Checkmate"
        elif self.board.is_stalemate():
            return "Stalemate"
        elif self.board.is_insufficient_material():
            return "Draw (Insufficient Material)"
        elif self.board.is_seventyfive_moves():
            return "Draw (75-move rule)"
        elif self.board.is_fivefold_repetition():
            return "Draw (Fivefold repetition)"
        return "Draw"

    def get_turn(self):
        return self.board.turn

    def set_fen(self, fen):
        self.board.set_fen(fen)
