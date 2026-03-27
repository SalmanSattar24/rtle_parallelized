"""
Phase 4 Simplified: Bilateral vs Baseline Comparison
Run this script in Colab: !python phase4_simple.py
"""

import sys
import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# Step 1-2: Setup and clone repo
repo_dir = "/content/rtle_parallelized"
if not os.path.exists(repo_dir):
    print("[CLONE] Cloning repository...")
    subprocess.run(["git", "clone", "https://github.com/SalmanSattar24/rtle_parallelized.git", repo_dir], check=True)
else:
    print("[PULL] Updating repository...")
    subprocess.run(["git", "-C", repo_dir, "pull"], capture_output=True)

# Step 3: Add paths and import
sys.path.insert(0, repo_dir)
from simulation.market_gym import Market
from rl_files.actor_critic import BilateralAgentLogisticNormal

print("[OK] Imports successful")
print(f"[INFO] Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# ============================================================================
# Step 4: Define baseline agent
# ============================================================================
class SymmetricFixedSpreadAgent:
    def __init__(self, action_dim=7):
        self.action = np.zeros(action_dim)
        self.action[1] = 1.0  # L1 only

    def get_action(self, obs):
        return self.action.copy()

# ============================================================================
# Step 5: Configuration
# ============================================================================
TRAIN_CONFIG = {
    'market_env': 'noise',
    'execution_agent': 'rl_agent',
    'volume': 40,
    'seed': 42,
    'terminal_time': 500,
    'time_delta': 50,
    'drop_feature': None,
    'inventory_max': 10,
    'penalty_weight': 1.0,
}

TRAIN_PARAMS = {
    'num_iterations': 200,
    'num_steps': 10,
    'learning_rate': 5e-4,
    'entropy_coef': 0.05,
}

EVAL_CONFIG = TRAIN_CONFIG.copy()
EVAL_CONFIG['seed'] = 100
EVAL_EPISODES = 1000

print("[OK] Config ready")

# ============================================================================
# Step 6: Create environment and agent with wrapper
# ============================================================================
print("\n[SETUP] Creating environment and agents...")

# Wrapper to adapt Market to have single_observation_space
class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

market_raw = Market(TRAIN_CONFIG)
obs, _ = market_raw.reset(seed=42)
market = EnvWrapper(market_raw)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilateral_agent = BilateralAgentLogisticNormal(market).to(device)
baseline_agent = SymmetricFixedSpreadAgent(TRAIN_CONFIG['volume'])

print(f"[OK] Bilateral agent on {device}")
print(f"[OK] Baseline agent ready")

# ============================================================================
# Step 7: Train bilateral agent (simplified)
# ============================================================================
print("\n[TRAIN] Training bilateral agent...")
print(f"Iterations: {TRAIN_PARAMS['num_iterations']}")

optimizer = torch.optim.Adam(bilateral_agent.parameters(), lr=TRAIN_PARAMS['learning_rate'], eps=1e-5)
training_returns = []

start_time = time.time()

for iteration in range(TRAIN_PARAMS['num_iterations']):
    bilateral_agent.variance = (0.32 - 1) * (iteration) / (TRAIN_PARAMS['num_iterations'] - 1) + 1

    obs, _ = market.reset(seed=42 + iteration)
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    episode_return = 0
    for step in range(TRAIN_PARAMS['num_steps']):
        with torch.no_grad():
            action, logprob, entropy, value = bilateral_agent.get_action_and_value(obs_batch)

        bid_action, ask_action = action
        env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())

        obs, reward, terminated, truncated, info = market.step(env_action)
        episode_return += reward

        obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        if terminated:
            break

    # Simple loss
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    _, logprob, entropy, value = bilateral_agent.get_action_and_value(obs_batch)

    reward_tensor = torch.tensor([episode_return], dtype=torch.float32).to(device)
    loss = -logprob.mean() * (reward_tensor - value).mean() - TRAIN_PARAMS['entropy_coef'] * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bilateral_agent.parameters(), 0.5)
    optimizer.step()

    training_returns.append(episode_return)

    if (iteration + 1) % 20 == 0:
        elapsed = time.time() - start_time
        avg = np.mean(training_returns[-20:])
        print(f"[{iteration+1:3d}] Return: {avg:7.3f} | Elapsed: {elapsed:6.1f}s")

print(f"[OK] Training complete!")

# ============================================================================
# Step 8: Evaluate bilateral
# ============================================================================
print(f"\n[EVAL] Evaluating bilateral agent ({EVAL_EPISODES} episodes)...")

bilateral_returns = []
bilateral_inventories = []

for ep in range(EVAL_EPISODES):
    eval_market_raw = Market(EVAL_CONFIG)
    eval_market = EnvWrapper(eval_market_raw)
    obs, _ = eval_market.reset(seed=100 + ep)

    ep_return = 0
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            bid_action, ask_action = bilateral_agent.deterministic_action(obs_tensor)
            env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())

        obs, reward, terminated, truncated, info = eval_market.step(env_action)
        ep_return += reward

        if terminated:
            break

    bilateral_returns.append(ep_return)
    bilateral_inventories.append(abs(info.get('net_inventory', 0)))

    if (ep + 1) % 200 == 0:
        print(f"  [{ep+1}/{EVAL_EPISODES}] Mean return: {np.mean(bilateral_returns):.4f}")

print(f"[OK] Bilateral mean return: {np.mean(bilateral_returns):.4f} +/- {np.std(bilateral_returns):.4f}")

# ============================================================================
# Step 9: Evaluate baseline
# ============================================================================
print(f"\n[EVAL] Evaluating baseline agent ({EVAL_EPISODES} episodes)...")

baseline_returns = []
baseline_inventories = []

for ep in range(EVAL_EPISODES):
    eval_market_raw = Market(EVAL_CONFIG)
    eval_market = EnvWrapper(eval_market_raw)
    obs, _ = eval_market.reset(seed=100 + ep)

    ep_return = 0
    while True:
        action = baseline_agent.get_action(obs)
        obs, reward, terminated, truncated, info = eval_market.step(action)
        ep_return += reward

        if terminated:
            break

    baseline_returns.append(ep_return)
    baseline_inventories.append(abs(info.get('net_inventory', 0)))

    if (ep + 1) % 200 == 0:
        print(f"  [{ep+1}/{EVAL_EPISODES}] Mean return: {np.mean(baseline_returns):.4f}")

print(f"[OK] Baseline mean return: {np.mean(baseline_returns):.4f} +/- {np.std(baseline_returns):.4f}")

# ============================================================================
# Step 10: Results
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS: BILATERAL vs BASELINE")
print("=" * 70)

bilateral_mean = np.mean(bilateral_returns)
baseline_mean = np.mean(baseline_returns)
improvement = bilateral_mean - baseline_mean
rel_improvement = (improvement / abs(baseline_mean)) * 100

print(f"\nBilateral:   {bilateral_mean:>10.4f} +/- {np.std(bilateral_returns):6.4f}")
print(f"Baseline:    {baseline_mean:>10.4f} +/- {np.std(baseline_returns):6.4f}")
print(f"Improvement: {improvement:>10.4f} ({rel_improvement:+.1f}%)")

if improvement > 0:
    print(f"\n[SUCCESS] Bilateral agent beats baseline by {improvement:.4f}!")
else:
    print(f"\n[INFO] Baseline better by {-improvement:.4f}")

print("=" * 70)

# ============================================================================
# Step 11: Plots
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].hist(bilateral_returns, bins=40, alpha=0.6, label='Bilateral', color='blue')
ax[0].hist(baseline_returns, bins=40, alpha=0.6, label='Baseline', color='orange')
ax[0].axvline(bilateral_mean, color='blue', linestyle='--', linewidth=2)
ax[0].axvline(baseline_mean, color='orange', linestyle='--', linewidth=2)
ax[0].set_xlabel('Return')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Return Distribution')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(np.cumsum(bilateral_returns), label='Bilateral', linewidth=2)
ax[1].plot(np.cumsum(baseline_returns), label='Baseline', linewidth=2)
ax[1].set_xlabel('Episode')
ax[1].set_ylabel('Cumulative Return')
ax[1].set_title('Cumulative Returns')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/content/phase4_results.png', dpi=100)
print("\n[OK] Plot saved to /content/phase4_results.png")

print("\n[COMPLETE] Phase 4 Simplified experiment finished!")
