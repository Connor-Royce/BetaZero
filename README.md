# BetaZero Intro

An AlphaZero-style AI for the game: Ultimate Tic-Tac-Toe (https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe), 
built as a deep dive into reinforcement learning and game AI. BetaZero learns entirely through self-play; meaning 
no handcrafted heuristics and no human game data, merely a neural network and a search algorithm figuring out the 
game together.

This is an ongoing project primarily focused on learning. Though the engine is functional and training, there is 
still plenty of room for improvements.

---

## How It Works

BetaZero follows the AlphaZero framework (AlphaZero is "a pioneering artificial intelligence developed by DeepMind
(Google) that mastered chess, shogi, and Go in a few hours of self-play without human knowledge"):

1. **Self-play** — The engine plays games against itself using MonteCarlo Tree Search (MCTS) guided by the neural network
2. **Training** — The resulting game data is used to improve the network
3. **Repeat** — The improved network guides better self-play, and so on

### Neural Network

The network takes a 5-channel 9×9 board encoding as input, and it outputs two things:

- **Policy head** — a probability distribution over all 81 possible moves, guiding MCTS toward promising moves and lines
- **Value head** — a scalar between -1 and 1 that estimates the current player's winning chances

The architecture uses an initial convolution layer followed by 4 residual blocks (64 channels each), then splits 
into the two heads. Residual connections help gradients flow during training and allow the network to learn more efficiently.

### Board Encoding (5 channels)

| Channel | Meaning |
|---|---|
| 0 | Current player's pieces |
| 1 | Opponent's pieces |
| 2 | Boards won by current player |
| 3 | Boards won by opponent |
| 4 | Legal moves mask |

### Monte Carlo Tree Search

MCTS builds a search tree by repeatedly:
- **Selecting** the most promising node via UCB score (balancing exploitation and exploration)
- **Expanding** leaf nodes using the network's policy as move priors
- **Evaluating** positions using the network's value head
- **Backpropagating** results up the tree

Move selection during self-play uses temperature scaling (T=1.0 for the first 30 moves for exploration) 
and Dirichlet noise at the root to ensure diverse training data.

---

## Project Structure

```
BetaZero/
├── game/
│   └── board.py              # UltimateBoard game logic
├── engines/
│   └── BetaZero/
│       ├── NeuralNetwork.py  # UltimateTTTNet (residual network)
│       ├── BetaZeroAgent.py  # BetaZero agent
│       ├── StateEncoder.py   # Board → tensor encoding/decoding
│       ├── MonteCarlo.py     # MCTS implementation
│       ├── Training/
│       │   ├── selfplay.py   # Self-play game generation & loss
│       │   └── train.py      # Outer training loop
│       └── models/           # Saved weights (not tracked by git)
│   └── random_agent.py       # Agent that plays random moves
├── UserInterface/
│   └── gui.py                # Tkinter desktop GUI
└── main.py
```

---

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- tkinter (usually bundled with Python)

### Play against BetaZero

```bash
python main.py
```

This opens the tkinter GUI. You can configure `agent_x` and `agent_o` in `main.py` to be human ("None"), 
the BetaZero AI ("BetaZeroAgent()"), or a random move generator ("RandomAgent()").

### Train from scratch

```bash
python engines/BetaZero/Training/train.py
```

Training will run self-play iterations and save weights to `engines/BetaZero/models/model_weights.pth` 
after each iteration. If weights already exist, training resumes from the last checkpoint by loading the 
already-existing weights.

GPU training is supported automatically, so if CUDA is available it will be used.

**Default training config:**
- 50 iterations
- 20 self-play games per iteration
- 100 MCTS simulations per move
- 64 channels, 4 residual blocks

---

## What remains to be done:

- **More training** — The network is still early in learning. More iterations and compute will make it
significantly stronger, as well as potential improvements for training efficiency.
- **Evaluation metric** — Adding a way to track the win rate against a RandomAgent across training will help
measure progress more concretely, and eventually tracking performance against online AIs that use minimax algorithms .
- **Hyperparameter tuning** — Experimenting with learning rate, MCTS simulation count, network depth/width, and other hyperparameters.
- **Model architecture improvements** — Potentially exploring attention mechanisms or larger networks
- **WebSocket bridge** — Connecting BetaZero to a browser-based GUI for a more polished interface

---

## Motivation

This project began for two main reasons. The first was because I wanted to create a model that could 
consistently beat humans in ultimate tic-tac-toe because I was very interested by this idea and thought 
it would be cool. The second reason is because although I understood some of the theory of neural networks, 
deep learning, and machine learning in general, I wanted to gain practical experience building something 
tangible with the concepts I have learned in the classroom, in books, and online. Because I am very interested 
in the game of Chess and am fascinated by the models behind the worlds top chess engines, I figured that this 
project would allow me to really understand how google's alphazero works. Ultimate Tic-Tac-Toe provided the 
perfect choice for this project because it is complex enough to be interesting, but not too large to where 
training the model would be impractical as a college student on basic hardware. With the help of Claude 
(primarily the opus 4.6 model), I was able to create a working engine that, once it has gone through significantly 
more training iterations, will be able to consistently beat human opponents.

This project is still a work in progress, and there is no doubt I have made many mistakes (or at least sub-optimal 
design choices) and still have much to learn. But the core engine is working and learning, and this project was 
immensely fun for me, and I enjoyed every second of it and everything that I learned along the way.


