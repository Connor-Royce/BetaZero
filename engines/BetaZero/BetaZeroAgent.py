
from pathlib import Path
from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.MonteCarlo import mcts_search
import torch

class BetaZeroAgent:
    def __init__(self):
        self.network = UltimateTTTNet(num_channels=64, num_res_blocks=4)
        model_path = Path(__file__).parent / "models" / "model_weights.pth"      
        self.network.load_state_dict(torch.load(model_path, weights_only=True))
        self.network.eval()
    def select_move(self,board):
        best_move, visits = mcts_search(board,self.network,add_noise=False)
        return best_move
