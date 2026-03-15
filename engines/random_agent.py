import random

class RandomAgent:
    def select_move(self, board):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)
