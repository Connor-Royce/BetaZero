import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import math
#from game.board import UltimateBoard
#from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.StateEncoder import encode_board, decode_policy


class MCTSNode:
    """
    A node in the MCTS search tree.
    """
    
    def __init__(self, board, parent=None, parent_action=None, prior_prob=0):
        """
        Args:
            board: UltimateBoard instance (the game state at this node)
            parent: Parent MCTSNode (None for root)
            parent_action: The move that led to this node
            prior_prob: Network's prior probability for this position
        """
        self.board = board
        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob
        
        self.children = {}  # Dictionary: {move: child_node}
        self.visit_count = 0.0
        self.total_value = 0.0
    
    def is_leaf(self):
        """Check if this node has no children yet."""
        return len(self.children) == 0
    
    def value(self):
        """Average value of this node (Q-value)."""
        if self.visit_count == 0:
            return 0
        return self.total_value/self.visit_count
    
    def select_favorite_child(self, c_puct=1.0):
        """
        Select the child with highest UCB score.
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            (move, child_node): The best child to explore
        """
        child_node_list = [((move,child),child.ucb_score(c_puct)) for move,child in self.children.items()]
        favorite_child = max(child_node_list, key=lambda x: x[1])
        return(favorite_child[0])



    def ucb_score(self, c_puct=1.0):
        """
        Calculate the UCB score for this node according to the following formula:

        UCB(node) = Q + c_puct * P * sqrt(parent_visits) / (1 + node_visits)
        Where:
        - Q = node.value() 
        - P = node.prior_prob 
        - c_puct = exploration constant  
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            float: UCB score
        """
        return self.value() + c_puct * self.prior_prob * (math.sqrt(self.parent.visit_count)/\
            (1+self.visit_count))
    
    def expand(self, network):
        """
        Expand this leaf node by adding children for all legal moves.
        
        Args:
            network: The neural network (UltimateTTTNet)
        
        Returns:
            The value the network outputs
        """

        legal_moves = self.board.get_legal_moves()
        state=encode_board(self.board)
        policy, value = network.predict(state)

        # Convert policy to move probabilities dict
        policy_tensor = torch.from_numpy(policy)
        move_probs = decode_policy(policy_tensor, self.board)
        
        for move in legal_moves:
            child_board = self.board.copy()
            child_board.apply_move(move)   
            prior_prob = float(move_probs[move])
            child_node = MCTSNode(child_board, parent=self, parent_action=move, prior_prob=prior_prob)
            self.children[move] = child_node
        
        return value

    def backpropagate(self, value):
        """
        Update this node and all ancestors with the evaluation result.
        
        Args:
            value: The value to propagate (from current player's perspective at leaf)
        
        Returns:
            None (modifies node statistics)
        """
        self.visit_count += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


def mcts_search(board, network, n=100, add_noise=False):
    """
    Run MCTS from the given board state.
    
    Args:
        board: UltimateBoard instance (current game state)
        network: UltimateTTTNet instance (for evaluation and priors)
        n: How many simulations to run
    
    Returns:
        best_move: The move (board_idx, cell_idx) to play
        visit_counts: A dictionary of the visit counts for all children
    """
    root = MCTSNode(board)
    root_value = root.expand(network)
    root.backpropagate(root_value)

    if add_noise:
        noise = np.random.dirichlet([0.03] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior_prob = 0.75 * child.prior_prob + 0.25 * noise[i]

    for i in range(n):
        # Starting at the root, repeatedly pick the best child (UCB) until it reaches a leaf
        node = root
        while not node.is_leaf():
            move, node = node.select_favorite_child()
        if node.board.is_terminal():
            result = node.board.check_macro()
            if result == 4:
                value = 0.0
            else:
                value = result * node.board.player # Convert absolute result to current player's perspective
        else:
            value = node.expand(network) # use the network to expand that leaf
        
        node.backpropagate(value) # Backpropagating: send the value back up the tree
        
    node_list = [((move,node),node.visit_count) for move,node in root.children.items()]
    best_move = max(node_list, key=lambda x: x[1])
    
    visit_counts = {move:node.visit_count for move,node in root.children.items()}

    return(best_move[0][0],visit_counts)
