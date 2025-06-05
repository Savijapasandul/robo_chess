import chess
from stockfish import Stockfish

class StockfishEngine:
    def __init__(self, path="/home/savija/projects/robo_chess/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"):
        self.engine = Stockfish(path=path, parameters={"Threads": 2, "Minimum Thinking Time": 30})
        self.skill_level = 1  # Default skill level
        self.engine.set_skill_level(self.skill_level)

    def set_fen(self, fen):
        self.engine.set_fen_position(fen)

    def get_best_move(self):
        return self.engine.get_best_move()

    def set_skill_level(self, level):
        """
        Set Stockfish skill level (0-20)
        """
        level = max(0, min(20, int(level)))
        self.engine.set_skill_level(level)
        self.skill_level = level

    def get_skill_level(self):
        """
        Return current skill level
        """
        return self.skill_level

    def close(self):
        pass  # Stockfish Python wrapper handles cleanup
