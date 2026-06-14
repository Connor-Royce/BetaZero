
from pathlib import Path
from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.MonteCarlo import mcts_search
import torch

class BetaZeroAgent:
    def __init__(self):
        self.network = UltimateTTTNet(num_channels=64, num_res_blocks=4)
        model_path = Path(__file__).parent / "Models" / "model_weights.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.network.load_state_dict(checkpoint['network'])
        self.network.to(device)
        self.network.eval()
    def select_move(self,board):
        best_move, visits = mcts_search(board,self.network,add_noise=False, n=1600)
        return best_move
