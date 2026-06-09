import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.Training.selfplay import batched_self_play,loss_computation,evaluate_vs_random,play_vs_random, generate_symmetries
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")
network = UltimateTTTNet(num_channels=64, num_res_blocks=4).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,600], gamma=0.1)
num_iterations = 1000

replay_buffer = deque(maxlen=800000)  # how many iterations of data to keep
batch_size = 64
num_epochs = 10

model_path = Path(__file__).parent.parent / "Models" / "model_weights.pth"

if model_path.exists():
    checkpoint = torch.load(model_path, weights_only=False)
    network.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_iteration = checkpoint['iteration'] + 1
    print(f"Resumed from iteration {start_iteration}")
else:
    print("No saved weights found, starting fresh")
    start_iteration = 0


for iteration in range(start_iteration, start_iteration + num_iterations):
    network.eval()
    iteration_data = []
    
    iteration_data = batched_self_play(network, num_games=50, num_simulations=400)

    for game_data in iteration_data:
        replay_buffer.extend(generate_symmetries(game_data))

    # mini-batch training — fewer epochs early to avoid overfitting noisy data
    network.train()
    buffer_list = list(replay_buffer)
    effective_epochs = min(num_epochs, 2 + iteration // 20)

    for epoch in range(effective_epochs):
        random.shuffle(buffer_list)
        chunk_list = [buffer_list[i:i + batch_size] for i in range(0, len(buffer_list), batch_size)]
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        for chunk in chunk_list:
            optimizer.zero_grad()
            loss, policy_loss, value_loss = loss_computation(chunk, network)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

    
    scheduler.step()

    log_line = f"Iteration {iteration} | Loss: {epoch_loss / len(chunk_list):.5f} | Policy: {epoch_policy_loss / len(chunk_list):.5f} | Value: {epoch_value_loss / len(chunk_list):.5f} | LR: {scheduler.get_last_lr()[0]}"
    if iteration % 5 == 0:
        win_rate = evaluate_vs_random(network, num_games=40, n=200)
        log_line += f" | vs Random: {win_rate:.1%}"
    with open("training_log.txt", "a") as f:
        f.write(log_line + "\n")
    print(log_line)

    #print(f"Iteration {iteration}, Avg Loss: {epoch_loss / len(chunk_list):.5f}")

    model_path.parent.mkdir(exist_ok=True)  # Creates the models/ folder if it doesn't exist
    torch.save({
    'network': network.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'iteration': iteration
}, model_path)
    print("Weights saved")

    # if iteration % 5 == 0:  # every N iterations
    #     evaluate_vs_random(network, num_games=40, n=50)
        # # Quick diagnostic: sample a few positions and check value predictions
        # sample = random.sample(buffer_list, min(5, len(buffer_list)))
        # network.eval()
        # for state, policy, true_val in sample:
        #     _, pred_val = network.predict(state)
        #     print(f"  true = {true_val:.1f}  pred = {pred_val[0]:.3f}")
    
    