import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
from game.board import UltimateBoard
from engines.BetaZero.StateEncoder import encode_board, decode_policy
from engines.BetaZero.MonteCarlo import MCTSNode


def _run_searches(network, boards, roots, indices, num_simulations, leaves_per_game):
    """
    Run batched MCTS for the given games, with every leaf evaluated by `network`.
    Same virtual-loss batching scheme as batched_self_play, but with no
    Dirichlet noise and no training-data collection (evaluation must measure
    the policy as-is).
    """
    device = next(network.parameters()).device

    # ---- Expand all root nodes (batched) ----
    states = [encode_board(boards[i]) for i in indices]
    state_batch = torch.from_numpy(np.stack(states)).float().to(device)
    with torch.no_grad():
        policy_logits, values = network(state_batch)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.cpu().numpy()

    for j, i in enumerate(indices):
        board = boards[i]
        root = roots[i]
        move_probs = decode_policy(torch.from_numpy(policies[j]), board)
        for move in board.get_legal_moves():
            child_board = board.copy()
            child_board.apply_move(move)
            root.children[move] = MCTSNode(child_board, parent=root,
                                           parent_action=move,
                                           prior_prob=float(move_probs[move]))
        root.backpropagate(float(values[j]))

    # ---- Simulations (batched with virtual loss) ----
    sims_done = 0
    while sims_done < num_simulations:
        batch_size_this_round = min(leaves_per_game, num_simulations - sims_done)

        leaves_to_evaluate = []
        states_to_evaluate = []
        terminal_leaves = []
        selected_this_round = set()

        for i in indices:
            for _ in range(batch_size_this_round):
                node = roots[i]
                while not node.is_leaf():
                    move, node = node.select_favorite_child()

                if node.board.is_terminal():
                    result = node.board.check_macro()
                    value = 0.0 if result == 4 else result * node.board.player
                    terminal_leaves.append((node, value))
                elif id(node) not in selected_this_round:
                    selected_this_round.add(id(node))
                    node.apply_virtual_loss()
                    leaves_to_evaluate.append(node)
                    states_to_evaluate.append(encode_board(node.board))

        for node, value in terminal_leaves:
            node.backpropagate(value)

        if states_to_evaluate:
            state_batch = torch.from_numpy(np.stack(states_to_evaluate)).float().to(device)
            with torch.no_grad():
                policy_logits, values = network(state_batch)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.cpu().numpy()

            for j, leaf in enumerate(leaves_to_evaluate):
                leaf.revert_virtual_loss()
                move_probs = decode_policy(torch.from_numpy(policies[j]), leaf.board)
                for move in leaf.board.get_legal_moves():
                    child_board = leaf.board.copy()
                    child_board.apply_move(move)
                    leaf.children[move] = MCTSNode(child_board, parent=leaf,
                                                   parent_action=move,
                                                   prior_prob=float(move_probs[move]))
                leaf.backpropagate(float(values[j]))

        sims_done += batch_size_this_round


def network_match(net_a, net_b, num_games=40, num_simulations=200,
                  leaves_per_game=8, temp_moves=6):
    """
    Play a head-to-head match between two networks with batched MCTS.

    net_a plays X (player 1) in even-numbered games and O in odd-numbered
    games so neither side keeps the first-move advantage. The first
    `temp_moves` plies of each game are sampled proportionally to visit
    counts so the match isn't the same two deterministic games repeated;
    after that, play is greedy.

    Returns:
        (a_wins, b_wins, draws)
    """
    boards = [UltimateBoard() for _ in range(num_games)]
    move_counters = [0] * num_games
    active = [True] * num_games
    # Which network plays player 1 / player -1 in each game
    nets = [{1: net_a, -1: net_b} if g % 2 == 0 else {1: net_b, -1: net_a}
            for g in range(num_games)]

    while any(active):
        active_indices = [i for i in range(num_games) if active[i]]

        # Fresh root per move, same as self-play (no tree reuse)
        roots = [None] * num_games
        for i in active_indices:
            roots[i] = MCTSNode(boards[i])

        # Group games by which network is to move, so each group shares
        # batched evaluations. The mover's own network evaluates every
        # leaf in its search, as in AlphaZero evaluation matches.
        group_a = [i for i in active_indices if nets[i][boards[i].player] is net_a]
        group_b = [i for i in active_indices if nets[i][boards[i].player] is net_b]

        if group_a:
            _run_searches(net_a, boards, roots, group_a, num_simulations, leaves_per_game)
        if group_b:
            _run_searches(net_b, boards, roots, group_b, num_simulations, leaves_per_game)

        for i in active_indices:
            children = roots[i].children
            moves = list(children.keys())
            counts = np.array([children[m].visit_count for m in moves], dtype=np.float32)

            if move_counters[i] < temp_moves:
                probs = counts / counts.sum()
                idx = np.random.choice(len(moves), p=probs)
            else:
                idx = int(np.argmax(counts))

            boards[i].apply_move(moves[idx])
            move_counters[i] += 1
            if boards[i].is_terminal():
                active[i] = False

    a_wins = b_wins = draws = 0
    for g in range(num_games):
        result = boards[g].check_macro()
        if result == 4:
            draws += 1
        elif nets[g][result] is net_a:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, draws


def elo_difference(score):
    """Convert a match score (0..1) into an Elo difference."""
    score = min(max(score, 0.01), 0.99)  # avoid infinities at 0% or 100%
    return 400 * math.log10(score / (1 - score))


def find_baseline_checkpoint(checkpoint_dir, current_iteration, gap):
    """
    Pick the snapshot to ladder against: the newest checkpoint at least
    `gap` iterations old. Early in training, when no checkpoint is that
    old yet, fall back to the oldest one as long as it is at least 25
    iterations behind; otherwise return None (skip the eval).
    """
    if not checkpoint_dir.exists():
        return None
    candidates = []
    for p in checkpoint_dir.glob("iter_*.pth"):
        try:
            candidates.append((int(p.stem.split("_")[1]), p))
        except ValueError:
            continue
    if not candidates:
        return None

    old_enough = [(it, p) for it, p in candidates if it <= current_iteration - gap]
    if old_enough:
        return max(old_enough)[1]

    earliest_it, earliest_p = min(candidates)
    if current_iteration - earliest_it >= 25:
        return earliest_p
    return None


if __name__ == "__main__":
    # Pit two saved checkpoints against each other:
    #   python evaluation.py <weights_a.pth> <weights_b.pth> [num_games] [num_simulations]
    # Accepts both ladder snapshots (raw state_dict) and the full training
    # checkpoint (dict with a 'network' key).
    from engines.BetaZero.NeuralNetwork import UltimateTTTNet

    path_a, path_b = sys.argv[1], sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    num_simulations = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_net(path):
        net = UltimateTTTNet(num_channels=64, num_res_blocks=4).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        net.load_state_dict(state['network'] if 'network' in state else state)
        net.eval()
        return net

    net_a = load_net(path_a)
    net_b = load_net(path_b)

    a_wins, b_wins, draws = network_match(net_a, net_b, num_games=num_games,
                                          num_simulations=num_simulations)
    score = (a_wins + 0.5 * draws) / num_games
    print(f"{Path(path_a).stem} vs {Path(path_b).stem}: "
          f"{a_wins}W/{b_wins}L/{draws}D  "
          f"score {score:.1%}  Elo {elo_difference(score):+.0f}")
