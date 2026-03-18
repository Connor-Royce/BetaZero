import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.Training.selfplay import self_play_game,loss_computation,evaluate_vs_random,play_vs_random
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")
network = UltimateTTTNet(num_channels=64, num_res_blocks=4).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=.0001)
num_iterations = 50

replay_buffer = deque(maxlen=10000)  # how many iterations of data to keep
batch_size = 64
num_epochs = 10

model_path = Path(__file__).parent.parent / "models" / "model_weights.pth"

if model_path.exists():
    checkpoint = torch.load(model_path, weights_only=False)
    network.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iteration = checkpoint['iteration'] + 1
    print(f"Resumed from iteration {start_iteration}")
else:
    print("No saved weights found, starting fresh")


for iteration in range(start_iteration, start_iteration + num_iterations):
    network.eval()
    iteration_data = []
    for i in range(10):
        iteration_data += self_play_game(network)
    for i in range(10):
        iteration_data += play_vs_random(network)

    # add to replay buffer instead of using directly
    replay_buffer.extend(iteration_data)
    # mini-batch training
    network.train()
    buffer_list = list(replay_buffer)

    for epoch in range(num_epochs):
        random.shuffle(buffer_list)
        chunk_list = [buffer_list[i:i + batch_size] for i in range(0, len(buffer_list), batch_size)]
        epoch_loss = 0.0
        for chunk in chunk_list:
            optimizer.zero_grad()
            loss = loss_computation(chunk, network)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()
    
    print(f"Iteration {iteration}, Avg Loss: {epoch_loss / len(chunk_list):.5f}")

    model_path.parent.mkdir(exist_ok=True)  # Creates the models/ folder if it doesn't exist
    torch.save({'network': network.state_dict(),'optimizer': optimizer.state_dict(),'iteration': iteration}, model_path)
    print("Weights saved")

    if iteration % 5 == 0:  # every N iterations
        evaluate_vs_random(network, num_games=40, n=50)
        # Quick diagnostic: sample a few positions and check value predictions
        sample = random.sample(buffer_list, min(5, len(buffer_list)))
        network.eval()
        for state, policy, true_val in sample:
            _, pred_val = network.predict(state)
            print(f"  true = {true_val:.1f}  pred = {pred_val[0]:.3f}")
    
    