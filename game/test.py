# Test script for test.py

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.board import UltimateBoard
from engines.BetaZero.StateEncoder import (
    encode_current_player_pieces,
    encode_opponent_pieces,
    encode_current_player_won_boards,
    encode_opponent_won_boards,
    convert_to_position,
    encode_legal_moves_mask,
    encode_board
)

def test_encoding():
    board = UltimateBoard()
    
    '''
    print("=== Initial State ===")
    print(f"Current player (to move): {board.player}")  # Should be 1 (X)
    
    # Make a simple move
    board.apply_move((0, 0))  # X plays top-left of board 0
    
    print("\n=== After X plays (0,0) ===")
    print(f"Current player (to move): {board.player}")  # Should be -1 (O)
    
    # Check encoding
    x_pieces = encode_current_player_pieces(board)
    o_pieces = encode_opponent_pieces(board)
    
    print(f"\nCurrent player pieces sum: {x_pieces.sum()}")  # Should be 0 (O has no pieces)
    print(f"Opponent pieces sum: {o_pieces.sum()}")  # Should be 1 (X has 1 piece)
    
    row, col = convert_to_position(0, 0)
    print(f"\nPosition (0,0) maps to grid ({row},{col})")
    print(f"Opponent channel at ({row},{col}): {o_pieces[row, col]}")  # Should be 1.0
    
    print("\n=== Testing Won Boards ===")
    
    # Win board 0 for X
    board2 = UltimateBoard()
    board2.apply_move((0, 0))  # X
    board2.apply_move((0, 1))  # O
    board2.apply_move((0, 4))  # X
    board2.apply_move((4, 2))  # O
    board2.apply_move((0, 8))  # X wins board 0 (diagonal 0,4,8)
    board2.apply_move((2, 4))  # O
    
    print(f"Macro board: {board2.macro}")  # board 0 should be 1 (won by X)
    print(f"Current player: {board2.player}")  # Should be -1 (O to move)
    
    won_boards = encode_opponent_won_boards(board2)
    print(f"\nWon boards channel sum: {won_boards.sum()}")  # Should be 9 if working correctly
    print("\nWon boards channel (should have 1s in top-left 3x3):")
    print(won_boards)
    legal_moves = encode_legal_moves_mask(board2)
    print(legal_moves)'''

    board.apply_move((4, 4))

    state = encode_board(board)
    print("State shape:", state.shape)  # Should be (5, 9, 9)
    print("Channel 0 sum (current player):", state[0].sum())
    print("Channel 1 sum (opponent):", state[1].sum())
    print("Channel 4 sum (legal moves):", state[4].sum())

if __name__ == "__main__":
    test_encoding()