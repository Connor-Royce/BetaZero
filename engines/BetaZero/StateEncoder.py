
import numpy as np

def convert_to_position(board_idx, cell_idx):
    """
    Convert board and cell indices to position in 9x9 grid.
    
    Args:
        board_idx: Which local board (0-8)
        cell_idx: Which cell in that board (0-8)
    
    Returns:
        (row, col): Position in 9x9 grid
    """
    
    big_row = board_idx // 3
    big_col = board_idx % 3
    
    cell_row = cell_idx // 3
    cell_col = cell_idx % 3
    
    # Formula for 9x9 coordinates:
    global_row = big_row * 3 + cell_row
    global_col = big_col * 3 + cell_col
    
    return (global_row, global_col)



def encode_current_player_pieces(board):
    """
    Create a 9x9 array showing where the player about to move has pieces.
    
    Args:
        board: UltimateBoard instance
    
    Returns:
        np.array of shape (9, 9)
    """
    channel = np.zeros((9, 9), dtype=np.float32)
    current_player = board.player
    
    for board_idx in range(9):
        for cell_idx in range(9):
            cell_value = board.boards[board_idx][cell_idx]
            
            if cell_value == current_player:
                row, col = convert_to_position(board_idx, cell_idx)
                channel[row, col] = 1.0
    
    return channel

def encode_opponent_pieces(board):
    """
    Create a 9x9 array showing where opponent (relative to current player) has pieces.
    
    Args:
        board: UltimateBoard instance
    
    Returns:
        np.array of shape (9, 9)
    """
    channel = np.zeros((9, 9), dtype=np.float32)
    opponent = -board.player
    
    for board_idx in range(9):
        for cell_idx in range(9):
            cell_value = board.boards[board_idx][cell_idx]
            
            if cell_value == opponent:
                row, col = convert_to_position(board_idx, cell_idx)
                channel[row, col] = 1.0
            
    return channel

def encode_current_player_won_boards(board):
    """
    Create a 9x9 array showing which local boards have been won by the current player (player about to move).
    
    Args:
        board: UltimateBoard instance
    
    Returns:
        np.array of shape (9, 9) with 1s in cells of boards won by current player (blocks of 3x3)
    """
    channel = np.zeros((9, 9), dtype=np.float32)
    current_player = board.player
    for board_idx in range(9):       
        if board.macro[board_idx] == current_player:
            for cell_idx in range(9):
                channel[convert_to_position(board_idx, cell_idx)] = 1.0
    return channel

def encode_opponent_won_boards(board):
    """
    Create a 9x9 array showing which local boards have been won by the opponent (player who just moved).
    
    Args:
        board: UltimateBoard instance
    
    Returns:
        np.array of shape (9, 9) with 1s in cells of boards won by opponent (blocks of 3x3)
    """
    channel = np.zeros((9, 9), dtype=np.float32)
    opponent = -board.player
    for board_idx in range(9):       
        if board.macro[board_idx] == opponent:
            for cell_idx in range(9):         
                channel[convert_to_position(board_idx, cell_idx)] = 1.0
    return channel

def encode_legal_moves_mask(board):
    """
    Create a 9x9 array showing what the legal moves are.
    
    Args:
        board: UltimateBoard instance
    
    Returns:
        np.array of shape (9, 9) with 1s in cells where moves are allowed.
    """
    channel = np.zeros((9, 9), dtype=np.float32)
    for move in board.get_legal_moves():
        row, col = move
        channel[convert_to_position(row, col)] = 1.0
    return channel

def encode_board(board):
    """
    Encode the full board state into a 5-channel tensor.
    
    Args:
        board: UltimateBoard instance
        
    Returns:
        np.array of shape (5, 9, 9)
    """
    channel0 = encode_current_player_pieces(board)
    channel1 = encode_opponent_pieces(board)
    channel2 = encode_current_player_won_boards(board)
    channel3 = encode_opponent_won_boards(board)
    channel4 = encode_legal_moves_mask(board)
    
    state = np.stack([channel0, channel1, channel2, channel3, channel4])
    
    return state

def decode_policy(policy_tensor, board):
    """
    Convert policy output tensor to move probabilities dictionary.
    
    Args:
        policy_tensor: torch.Tensor or np.array of shape (81,) with probabilities
        board: UltimateBoard instance to get legal moves
        
    Returns:
        dict mapping (board_idx, cell_idx) -> probability
    """
    moves = board.get_legal_moves()
    result = {}
    for move in moves:
        row, col = convert_to_position(move[0],move[1])
        flat_index = (row * 9) + col
        result[move] = policy_tensor[flat_index]
    return result



    