import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import time
import torch
from engines.BetaZero.NeuralNetwork import UltimateTTTNet
from engines.BetaZero.Training.selfplay import batched_self_play,loss_computation,evaluate_vs_random,play_vs_random, generate_symmetries
from engines.BetaZero.Training.evaluation import network_match, find_baseline_checkpoint, elo_difference
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
batches_per_iteration = 1000

model_path = Path(__file__).parent.parent / "Models" / "model_weights.pth"
checkpoint_dir = model_path.parent / "checkpoints"  # frozen snapshots for ladder evals
CHECKPOINT_INTERVAL = 25  # save a frozen snapshot this often
EVAL_GAP = 100  # ladder-eval against the snapshot from ~this many iterations back

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
    iter_start = time.time()
    network.eval()
    iteration_data = []
    
    iteration_data = batched_self_play(network, num_games=50, num_simulations=400)

    for game_data in iteration_data:
        replay_buffer.extend(generate_symmetries(game_data))

    # mini-batch training — sample a fixed number of random minibatches per
    # iteration instead of full passes over the buffer, so training cost stays
    # constant and old samples get a bounded number of gradient views
    network.train()
    buffer_list = list(replay_buffer)

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    for _ in range(batches_per_iteration):
        batch = random.sample(buffer_list, min(batch_size, len(buffer_list)))
        optimizer.zero_grad()
        loss, policy_loss, value_loss = loss_computation(batch, network)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),max_norm=1)
        optimizer.step()
        total_loss_sum += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    scheduler.step()

    log_line = f"Iteration {iteration} | Loss: {total_loss_sum / batches_per_iteration:.5f} | Policy: {policy_loss_sum / batches_per_iteration:.5f} | Value: {value_loss_sum / batches_per_iteration:.5f} | LR: {scheduler.get_last_lr()[0]}"
    # if iteration % 25 == 0:
    #     network.eval()  
    #     win_rate = evaluate_vs_random(network, num_games=40, n=200)
    #     log_line += f" | vs Random: {win_rate:.1%}"

    '''Ladder evaluation: play the current network against a frozen snapshot
    from ~EVAL_GAP iterations ago. A score that stays near 50% for several
    evals in a row means self-improvement has stalled.'''
    if iteration % CHECKPOINT_INTERVAL == 0:
        baseline = find_baseline_checkpoint(checkpoint_dir, iteration, EVAL_GAP)
        if baseline is not None:
            network.eval()
            old_net = UltimateTTTNet(num_channels=64, num_res_blocks=4).to(device)
            old_net.load_state_dict(torch.load(baseline, map_location=device, weights_only=True))
            old_net.eval()
            wins, match_losses, draws = network_match(network, old_net, num_games=40, num_simulations=200)
            score = (wins + 0.5 * draws) / (wins + match_losses + draws)
            log_line += f" | vs {baseline.stem}: {wins}W/{match_losses}L/{draws}D (Elo {elo_difference(score):+.0f})"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), checkpoint_dir / f"iter_{iteration:05d}.pth")

    log_line += f" | {time.time() - iter_start:.0f}s"
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

    
    