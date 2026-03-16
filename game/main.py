import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.board import UltimateBoard
from engines.random_agent import RandomAgent
from engines.BetaZero.BetaZeroAgent import BetaZeroAgent
from UserInterface.gui import UltimateTTTGUI

def main():
    board = UltimateBoard()
    
    try:
        #gui = UltimateTTTGUI(board, agent_x=None, agent_o=BetaZeroAgent())
        #gui = UltimateTTTGUI(board, agent_x=RandomAgent(), agent_o=BetaZeroAgent())
        gui = UltimateTTTGUI(board, agent_x=BetaZeroAgent(), agent_o=RandomAgent())

        gui.run()  # Start the game loop
    except Exception as e:
        print(f"Error starting game: {e}")

if __name__ == "__main__":
    main()

    
