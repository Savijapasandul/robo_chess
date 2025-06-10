import chess
import chess.engine
import random

class ChessLogic:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.engine = None
        self.ai_difficulty = 1
        
    def start_engine(self, engine_path):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        except:
            self.engine = None
    
    def make_move(self, move, promotion=None):
        if promotion:
            move.promotion = promotion
        self.board.push(move)
        self.move_history.append(move)
        return self.board
    
    def undo_move(self):
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            return True
        return False
    
    def reset_game(self):
        self.board.reset()
        self.move_history = []
        return self.board
    
    def get_legal_moves(self, square):
        return [move.to_square for move in self.board.legal_moves if move.from_square == square]
    
    def is_pawn_promotion(self, move):
        piece = self.board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if move.to_square in chess.SquareSet(chess.BB_BACKRANKS):
                return True
        return False
    
    def make_ai_move(self):
        if not self.engine:
            return self.make_random_move()
        
        try:
            think_time = 0.5 * self.ai_difficulty
            result = self.engine.play(
                self.board, 
                chess.engine.Limit(time=think_time),
                options={"Skill Level": self.ai_difficulty * 5}
            )
            return self.make_move(result.move)
        except:
            return self.make_random_move()
    
    def make_random_move(self):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            return self.make_move(move)
        return self.board
    
    def check_game_over(self):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            return "Stalemate! Game is a draw."
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material!"
        elif self.board.is_seventyfive_moves():
            return "Draw by 75-move rule!"
        elif self.board.is_fivefold_repetition():
            return "Draw by repetition!"
        return None
    
    def get_board_state(self):
        return self.board
    
    def set_difficulty(self, difficulty):
        self.ai_difficulty = difficulty
    
    def close_engine(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass