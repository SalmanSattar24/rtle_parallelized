"""
Quick training diagnostic - run in Colab to test performance
This version has reduced episodes and short episodes for fast testing
"""

import sys
import os
import numpy as np
import torch
import time

# Assuming this is run from the repo root
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "simulation"))
sys.path.insert(0, os.path.join(os.getcwd(), "rl_files"))

from simulation.market_gym import Market
from rl_files.actor_critic import BilateralAgentLogisticNormal

print("="*70)
print("QUICK TRAINING DIAGNOSTIC")
print("="*70)

# REDUCED config for fast testing
QUICK_CONFIG = {
    'market_env': 'noise',
    'execution_agent': 'rl_agent',
    'volume': 40,
    'seed': 42,
    'terminal_time': 100,  # REDUCED from 500
    'time_delta': 50,
    'drop_feature': None,
    'inventory_max': 10,
    'penalty_weight': 1.0,
}

QUICK_PARAMS = {
    'num_iterations': 5,   # REDUCED from 200
    'num_steps': 2,        # REDUCED from 10
    'learning_rate': 5e-4,
    'entropy_coef': 0.05,
    'vf_coef': 0.5,
    'gamma': 1.0,
}

# Setup
print("\n[SETUP] Creating environment and agent...")
market_env = Market(QUICK_CONFIG)
obs, _ = market_env.reset(seed=42)

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    def step(self, action):
        return self.env.step(action)

market = EnvWrapper(market_env)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilateral_agent = BilateralAgentLogisticNormal(market).to(device)
print(f"[OK] Using device: {device}")

# DIAGNOSTIC: Test single episode
print("\n[DIAG] Testing single episode...")
obs, _ = market.reset(seed=42)
ep_start = time.time()
timesteps = 0

while True:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        actions, log_prob, entropy, value = bilateral_agent.get_action_and_value(obs_tensor)

    bid_action, ask_action = actions
    env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
    obs, reward, terminated, truncated, info = market.step(env_action)
    timesteps += 1

    if terminated:
        break

ep_time = time.time() - ep_start
print(f"✓ Episode: {timesteps} steps in {ep_time:.2f}s")
print(f"  Time per step: {ep_time/timesteps:.3f}s")
print(f"  Est. training time (5 iter × 2 steps × {timesteps} steps): {5*2*ep_time:.1f}s")

# TRAINING: Quick loop
print("\n[TRAIN] Starting quick training...")
print("-"*70)

optimizer = torch.optim.Adam(bilateral_agent.parameters(), lr=QUICK_PARAMS['learning_rate'])
training_returns = []

start_train = time.time()

for iteration in range(QUICK_PARAMS['num_iterations']):
    iter_start = time.time()
    batch_rewards = []
    batch_states = []
    batch_actions = []
    batch_values = []
    batch_log_probs = []

    for step in range(QUICK_PARAMS['num_steps']):
        obs, _ = market.reset(seed=42 + iteration * QUICK_PARAMS['num_steps'] + step)
        ep_return = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                actions, log_prob, entropy, value = bilateral_agent.get_action_and_value(obs_tensor)

            batch_states.append(obs_tensor.detach())
            batch_actions.append(actions)
            batch_log_probs.append(log_prob.detach())
            batch_values.append(value.detach())

            bid_action, ask_action = actions
            env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
            obs, reward, terminated, truncated, info = market.step(env_action)

            batch_rewards.append(reward)
            ep_return += reward

            if terminated:
                break

        training_returns.append(ep_return)

    # Compute advantages
    batch_returns = []
    cumulative_return = 0
    for i in range(len(batch_rewards) - 1, -1, -1):
        cumulative_return = batch_rewards[i] + QUICK_PARAMS['gamma'] * cumulative_return
        batch_returns.insert(0, cumulative_return)

    batch_advantages = [batch_returns[i] - batch_values[i].item() for i in range(min(len(batch_returns), len(batch_values)))]

    # Update
    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32).to(device)
    advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32).to(device)

    optimizer.zero_grad()
    total_loss = 0
    for i in range(min(len(batch_states), len(batch_advantages))):
        _, log_prob, entropy, value = bilateral_agent.get_action_and_value(batch_states[i], action=batch_actions[i])
        actor_loss = -(log_prob * advantages_tensor[i])
        value_loss = 0.5 * (value.squeeze() - returns_tensor[i]) ** 2
        entropy_bonus = -entropy * QUICK_PARAMS['entropy_coef']
        total_loss = total_loss + actor_loss + QUICK_PARAMS['vf_coef'] * value_loss + entropy_bonus

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(bilateral_agent.parameters(), max_norm=0.5)
    optimizer.step()

    iter_time = time.time() - iter_start
    avg_return = np.mean(training_returns[-QUICK_PARAMS['num_steps']:])

    print(f"[{iteration+1}/{QUICK_PARAMS['num_iterations']}] Time: {iter_time:.1f}s | Return: {avg_return:.2f}")

total_time = time.time() - start_train

print("-"*70)
print(f"✓ Quick training complete in {total_time:.1f}s")
print(f"\nFull training estimate (200 iterations × 10 steps):")
print(f"  Time per iteration: ~{total_time / QUICK_PARAMS['num_iterations']:.1f}s")
print(f"  Est. total: ~{(200*10)/(QUICK_PARAMS['num_iterations']*QUICK_PARAMS['num_steps']) * total_time / 60:.1f} minutes")
print("\n" + "="*70)
