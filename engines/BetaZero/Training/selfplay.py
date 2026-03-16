import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from game.board import UltimateBoard
#from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.StateEncoder import encode_board, convert_to_position
from engines.BetaZero.MonteCarlo import mcts_search
from engines.random_agent import RandomAgent


def self_play_game(network, n=100, temperature=1.0):
    """
    Play one full game using MCTS, recording training data.
    
    Args:
        network: UltimateTTTNet instance
        n: MCTS simulations per move
    
    Returns:
        training_data: List of (state, policy, value) tuples
    """
    board = UltimateBoard()
    game_history = []  # Store (state, policy, current_player) during the game
    move_counter = 0
    while not board.is_terminal():      
        # Run MCTS to get a move and visit counts
        best_move, visits = mcts_search(board, network, n,  add_noise=True)

        if move_counter < 30:
            moves = list(visits.keys())
            counts = np.array([visits[m] for m in moves], dtype=np.float32)
            counts = counts ** (1.0 / temperature)
            counts /= counts.sum()  # normalize to probabilities

            # Build policy from temperature-adjusted counts
            policy = np.zeros(81, dtype=np.float32)
            for i, move in enumerate(moves):
                row, col = convert_to_position(move[0], move[1])
                flat_index = (row * 9) + col
                policy[flat_index] = counts[i]
            
            state = encode_board(board)
            game_history.append((state, policy, board.player))

            idx = np.random.choice(len(moves), p=counts)
            board.apply_move(moves[idx])
        else:
            #Convert visit counts into a policy (probability distribution over 81 cells)
            policy = np.zeros(81, dtype=np.float32)
            for move, visit_count in visits.items():
                row, col = convert_to_position(move[0],move[1])
                flat_index = (row * 9) + col
                policy[flat_index] = visit_count
            policy /= np.sum(policy)  # Normalize to get probabilities
            state = encode_board(board)

            # Save (state, policy, board.player) to game_history
            game_history.append((state, policy, board.player))
            board.apply_move(best_move)
    
        move_counter += 1
 
    # Game is over — check the result
    result = board.check_macro()
    
    # Go back through game_history and assign values based on who actually won, from each position's perspective
    training_data = []
    for state, policy, player in game_history:
        if result == 4:
            value = 0.0  # Draw
        elif result == player:
            value = 1.0  # Current player won
        else:
            value = -1.0 # Current player lost
        training_data.append((state, policy, value))
    
    return training_data

def loss_computation(training_data_list,network):
    """
    computes loss of network's predictions
    Args: 
        training_data_batch: batch of training data
        network: neural network
    Returns: 
        total_loss: policy loss + value loss
    
    """
    states = []
    policies = []
    values = []
    for state, policy, value in training_data_list:
        states.append(state)                    
        policies.append(policy)               
        values.append(value)
    
    # Stack into tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for GPU training
    state_batch = torch.from_numpy(np.stack(states)).float().to(device)      # (B, C, 9, 9)
    policy_batch = torch.from_numpy(np.stack(policies)).float().to(device)    # (B, 81)
    value_batch = torch.from_numpy(np.array(values)).float().unsqueeze(1).to(device)   # (B, 1)

    # Network predicts from the state
    predicted_policy, predicted_value = network(state_batch)

    # Compare to the "answer-key" from self-play
    policy = torch.nn.functional.log_softmax(predicted_policy, dim=1)
    policy_loss = -torch.mean(torch.sum(policy_batch*policy,dim=1))
    value_loss = torch.nn.functional.mse_loss(predicted_value, value_batch)

    # Total loss tells PyTorch how wrong the network was
    total_loss = policy_loss + value_loss

    return total_loss

def evaluate_vs_random(network, num_games=40, n=50):
    """
    Play games against RandomAgent to measure strength.
    
    Args:
        network: UltimateTTTNet instance
        num_games: total games to play (half as X, half as O)
        n: MCTS simulations per move (can be lower than training for speed)
    
    Returns:
        win_rate: float between 0 and 1
    """
    
    wins = 0
    losses = 0
    draws = 0
    random = RandomAgent()
    for game_num in range(num_games):
        board = UltimateBoard()
        
        if game_num % 2 == 0:
            betazero_player = -1
        else:
            betazero_player = 1
        
        while not board.is_terminal():
            if board.player == betazero_player:
                best_move,visit_counts = mcts_search(board,network,n,add_noise=False)
                board.apply_move(best_move)
            else:
                random_move = random.select_move(board)
                board.apply_move(random_move)
        
        if board.is_terminal():
            if board.check_macro() == betazero_player:
                wins +=1
            elif board.check_macro() == -betazero_player:
                losses +=1
            else:
                draws += 1
        
    print(f"vs Random: {wins}W / {losses}L / {draws}D  ({wins/num_games:.1%} win rate)")
    return wins / num_games