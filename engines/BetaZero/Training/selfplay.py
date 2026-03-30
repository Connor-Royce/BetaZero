import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from game.board import UltimateBoard
from engines.BetaZero.StateEncoder import encode_board, convert_to_position, decode_policy
from engines.BetaZero.MonteCarlo import mcts_search,  MCTSNode 
from engines.random_agent import RandomAgent
import random as rando

def batched_self_play(network, num_games=50, num_simulations=400, temperature=1.0):
    """
    Play multiple games in parallel with batched neural network evaluation.
    
    Returns:
        all_training_data: list of (state, policy, value) tuples from all games
    """
    
    # ---- Phase 1: Initialize all games ----
    boards = [UltimateBoard() for _ in range(num_games)]
    roots = [MCTSNode(boards[i]) for i in range(num_games)]
    game_histories = [[] for _ in range(num_games)]
    active = [True] * num_games
    move_counters = [0] * num_games

    # ---- Main loop: keep playing until all games are done ----
    while any(active):

        # ---- Phase 2: Expand all root nodes (batched) ----
        states = []
        active_indices = []
        for i in range(num_games):
            if not active[i]:
                continue
            states.append(encode_board(boards[i]))
            active_indices.append(i)
        
        if not active_indices:
            break

        state_batch = torch.from_numpy(np.stack(states)).float().to(
            next(network.parameters()).device
        )
        with torch.no_grad():
            policy_logits, values = network(state_batch)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()

        for j, i in enumerate(active_indices):
            policy = policies[j]
            value = float(values[j])

            # Expand root using the batched results
            legal_moves = boards[i].get_legal_moves()
            move_probs = decode_policy(torch.from_numpy(policy), boards[i])

            for move in legal_moves:
                child_board = boards[i].copy()
                child_board.apply_move(move)
                prior = float(move_probs[move])
                child_node = MCTSNode(child_board, parent=roots[i], parent_action=move, prior_prob=prior)
                roots[i].children[move] = child_node

            roots[i].backpropagate(value)

            # Add Dirichlet noise to root children
            noise = np.random.dirichlet([0.03] * len(roots[i].children))
            for k, child in enumerate(roots[i].children.values()):
                child.prior_prob = 0.75 * child.prior_prob + 0.25 * noise[k]


        # ---- Phase 3: MCTS simulations (batched) ----
        for sim in range(num_simulations):

            # -- Step A: Selection --
            leaves_to_evaluate = []
            states_to_evaluate = []

            for i in active_indices:
                node = roots[i]
                while not node.is_leaf():
                    move, node = node.select_favorite_child()

                # Terminal leaf — backpropagate immediately
                if node.board.is_terminal():
                    result = node.board.check_macro()
                    if result == 4:
                        value = 0.0
                    else:
                        value = result * node.board.player
                    node.backpropagate(value)
                    continue

                # Non-terminal leaf — queue for batched evaluation
                leaves_to_evaluate.append((i, node))
                states_to_evaluate.append(encode_board(node.board))

            # -- Step B: Batched evaluation --
            if not states_to_evaluate:
                continue

            state_batch = torch.from_numpy(np.stack(states_to_evaluate)).float().to(
                next(network.parameters()).device
            )
            with torch.no_grad():
                policy_logits, values = network(state_batch)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.cpu().numpy()

            # -- Step C: Expansion and backpropagation --
            for j, (game_idx, leaf_node) in enumerate(leaves_to_evaluate):
                policy = policies[j]
                value = float(values[j])

                legal_moves = leaf_node.board.get_legal_moves()
                move_probs = decode_policy(torch.from_numpy(policy), leaf_node.board)

                for move in legal_moves:
                    child_board = leaf_node.board.copy()
                    child_board.apply_move(move)
                    prior = float(move_probs[move])
                    child_node = MCTSNode(child_board, parent=leaf_node, parent_action=move, prior_prob=prior)
                    leaf_node.children[move] = child_node

                leaf_node.backpropagate(value)


        # ---- Phase 4: Move selection ----
        for i in active_indices:
            visit_counts = {move: child.visit_count for move, child in roots[i].children.items()}
            moves = list(visit_counts.keys())
            counts = np.array([visit_counts[m] for m in moves], dtype=np.float32)

            # Temperature-based selection for early moves, greedy for later
            if move_counters[i] < 30:
                counts_temp = counts ** (1.0 / temperature)
                probs = counts_temp / counts_temp.sum()
            else:
                # Greedy — put all probability on the most visited move
                probs = np.zeros_like(counts)
                probs[np.argmax(counts)] = 1.0

            # Build policy target from visit counts (always use raw counts, not temperature-adjusted)
            policy_target = np.zeros(81, dtype=np.float32)
            total_visits = counts.sum()
            for j, move in enumerate(moves):
                row, col = convert_to_position(move[0], move[1])
                flat_index = row * 9 + col
                policy_target[flat_index] = counts[j] / total_visits

            # Record training data
            state = encode_board(boards[i])
            game_histories[i].append((state, policy_target, boards[i].player))

            # Sample move
            idx = np.random.choice(len(moves), p=probs)
            chosen_move = moves[idx]
            boards[i].apply_move(chosen_move)
            move_counters[i] += 1

            # Check if game is over
            if boards[i].is_terminal():
                active[i] = False
            else:
                # Create new root for next move
                roots[i] = MCTSNode(boards[i])


    # ---- Phase 5: Assign game outcomes ----
    all_training_data = []
    for i in range(num_games):
        result = boards[i].check_macro()
        for state, policy, player in game_histories[i]:
            if result == 4:
                value = 0.0
            elif result == player:
                value = 1.0
            else:
                value = -1.0
            all_training_data.append((state, policy, value))

    return all_training_data

# def self_play_game(network, n=400, temperature=1.0):
#     """
#     Play one full game using MCTS, recording training data.
    
#     Args:
#         network: UltimateTTTNet instance
#         n: MCTS simulations per move
    
#     Returns:
#         training_data: List of (state, policy, value) tuples
#     """
#     board = UltimateBoard()
#     game_history = []  # Store (state, policy, current_player) during the game
#     move_counter = 0
#     while not board.is_terminal():      
#         # Run MCTS to get a move and visit counts
#         best_move, visits = mcts_search(board, network, n,  add_noise=True)

#         if move_counter < 30:
#             moves = list(visits.keys())
#             counts = np.array([visits[m] for m in moves], dtype=np.float32)
#             counts = counts ** (1.0 / temperature)
#             counts /= counts.sum()  # normalize to probabilities

#             # Build policy from temperature-adjusted counts
#             policy = np.zeros(81, dtype=np.float32)
#             for i, move in enumerate(moves):
#                 row, col = convert_to_position(move[0], move[1])
#                 flat_index = (row * 9) + col
#                 policy[flat_index] = counts[i]
            
#             state = encode_board(board)
#             game_history.append((state, policy, board.player))

#             idx = np.random.choice(len(moves), p=counts)
#             board.apply_move(moves[idx])
#         else:
#             #Convert visit counts into a policy (probability distribution over 81 cells)
#             policy = np.zeros(81, dtype=np.float32)
#             for move, visit_count in visits.items():
#                 row, col = convert_to_position(move[0],move[1])
#                 flat_index = (row * 9) + col
#                 policy[flat_index] = visit_count
#             policy /= np.sum(policy)  # Normalize to get probabilities
#             state = encode_board(board)

#             # Save (state, policy, board.player) to game_history
#             game_history.append((state, policy, board.player))
#             board.apply_move(best_move)
    
#         move_counter += 1
 
#     # Game is over — check the result
#     result = board.check_macro()
    
#     # Go back through game_history and assign values based on who actually won, from each position's perspective
#     training_data = []
#     for state, policy, player in game_history:
#         if result == 4:
#             value = 0.0  # Draw
#         elif result == player:
#             value = 1.0  # Current player won
#         else:
#             value = -1.0 # Current player lost
#         training_data.append((state, policy, value))
    
#     return training_data

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

def play_vs_random(network, n=100, temperature=1.0):
    """
    Play one game of BetaZero vs RandomAgent, recording training data
    only from BetaZero's perspective.
    
    Args:
        network: UltimateTTTNet instance
        n: MCTS simulations per move
        temperature: temperature for move selection
    
    Returns:
        training_data: List of (state, policy, value) tuples
    """
    board = UltimateBoard()
    random_agent = RandomAgent()
    game_history = []  # Only store BetaZero's positions
    move_counter = 0
    
    betazero_player = rando.choice([-1,1]) 
    
    while not board.is_terminal():
        if board.player == betazero_player:
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
        else:
            random_move = random_agent.select_move(board)
            board.apply_move(random_move)     
        move_counter += 1
 
    result = board.check_macro()
    training_data = []
    for state, policy, player in game_history:
        if result == 4:
            value = 0.0
        elif result == betazero_player:
            value = 1.0
        else:
            value = -1.0 
        training_data.append((state, policy, value))
    
    return training_data

def generate_symmetries(info_tuple):
    """
    A function that generates the symmetries (since ultimate tic-tac-toe is invariant 
    to rotations and reflections)
    Since the board is a square composed of 9 smaller squares (the dihedral group D8,) 
    the function generates 8 symmetries

    Args:
        info_tuple: a (state,policy,value) tuple)

    Returns:
        symmetries: a list of 8 (state, policy, value) tuples
        
    """
    symmetries = []
   
    value = info_tuple[2]
    state_identity = info_tuple[0]
    policy_identity = info_tuple[1]
    policy_identity_2d = np.reshape(a=policy_identity,newshape=(9,9))
    single_state_reflection = np.flip(state_identity,axis=1)
    single_policy_reflection = np.flip(policy_identity_2d,axis=0)
    single_policy_reflection_flat = np.reshape(a=single_policy_reflection,newshape=(81,))
    symmetries.append((state_identity,policy_identity,value))
    symmetries.append((single_state_reflection,single_policy_reflection_flat,value))
    for i in range(1,4):
        # rotations (r,r^2,r^3)
        state_rotation=np.rot90(state_identity,axes=(1, 2), k=i)
        policy_rotation=np.rot90(policy_identity_2d,axes=(0, 1), k=i)
        policy_rotation_flat = policy_rotation.reshape(81)
        symmetries.append((state_rotation,policy_rotation_flat,value))

        #reflections (sr,sr^2,sr^3)
        state_reflection = np.flip(state_rotation,axis=1)
        policy_reflection = np.flip(policy_rotation,axis=0)
        policy_reflection_flat = policy_reflection.reshape(81)
        symmetries.append((state_reflection,policy_reflection_flat,value))
            
    return symmetries