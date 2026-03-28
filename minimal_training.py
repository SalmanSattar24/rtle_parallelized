"""Minimal training script - copy this into a Colab cell to replace cell 19"""

print("="*70)
print("TRAINING BILATERAL AGENT (MINIMAL VERSION)")
print("="*70)
print()

# REDUCED config
TRAIN_CONFIG_QUICK = {
    'market_env': 'noise',
    'execution_agent': 'rl_agent',
    'volume': 40,
    'seed': 42,
    'terminal_time': 50,  # Much shorter episodes
    'time_delta': 50,
    'drop_feature': None,
    'inventory_max': 10,
    'penalty_weight': 1.0,
}

TRAIN_PARAMS_QUICK = {
    'num_iterations': 10,  # Just 10 iterations for testing
    'num_steps': 2,
    'learning_rate': 5e-4,
    'entropy_coef': 0.05,
    'vf_coef': 0.5,
    'gamma': 1.0,
}

print(f"[CONFIG] Terminal time: {TRAIN_CONFIG_QUICK['terminal_time']}")
print(f"[CONFIG] Iterations: {TRAIN_PARAMS_QUICK['num_iterations']}")
print()

# Create fresh market for training (to avoid reusing old one)
market_quick = Market(TRAIN_CONFIG_QUICK)
optimizer = torch.optim.Adam(bilateral_agent.parameters(), lr=TRAIN_PARAMS_QUICK['learning_rate'])

training_returns = []
training_losses = []
start_time = time.time()

for iteration in range(TRAIN_PARAMS_QUICK['num_iterations']):
    print(f"[{iteration+1:2d}/{TRAIN_PARAMS_QUICK['num_iterations']}] Starting iteration...")
    
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_values = []
    batch_log_probs = []
    batch_entropies = []

    for step in range(TRAIN_PARAMS_QUICK['num_steps']):
        print(f"  Step {step+1}/{TRAIN_PARAMS_QUICK['num_steps']}: reset...", end='', flush=True)
        obs, _ = market_quick.reset(seed=42 + iteration * TRAIN_PARAMS_QUICK['num_steps'] + step)
        ep_return = 0
        timesteps = 0
        
        print(f" run...", end='', flush=True)
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                actions, log_prob, entropy, value = bilateral_agent.get_action_and_value(obs_tensor)

            batch_states.append(obs_tensor.detach())
            batch_actions.append(actions)
            batch_log_probs.append(log_prob.detach())
            batch_entropies.append(entropy.detach())
            batch_values.append(value.detach())

            bid_action, ask_action = actions
            env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
            obs, reward, terminated, truncated, info = market_quick.step(env_action)

            batch_rewards.append(reward)
            ep_return += reward
            timesteps += 1

            if terminated:
                break

        training_returns.append(ep_return)
        print(f" done ({timesteps} steps, return={ep_return:.2f})", flush=True)

    print(f"  Computing advantages and backprop...", end='', flush=True)
    
    # Compute advantages
    batch_returns = []
    batch_advantages = []
    cumulative_return = 0
    for i in range(len(batch_rewards) - 1, -1, -1):
        cumulative_return = batch_rewards[i] + TRAIN_PARAMS_QUICK['gamma'] * cumulative_return
        batch_returns.insert(0, cumulative_return)
        if i < len(batch_values):
            advantage = cumulative_return - batch_values[i].item()
            batch_advantages.insert(0, advantage)

    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32).to(device)
    advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32).to(device)

    optimizer.zero_grad()
    total_loss = 0
    for i in range(min(len(batch_states), len(batch_advantages))):
        _, log_prob, entropy, value = bilateral_agent.get_action_and_value(batch_states[i], action=batch_actions[i])
        actor_loss = -(log_prob * advantages_tensor[i])
        value_loss = 0.5 * (value.squeeze() - returns_tensor[i]) ** 2
        entropy_bonus = -entropy * TRAIN_PARAMS_QUICK['entropy_coef']
        total_loss = total_loss + actor_loss + TRAIN_PARAMS_QUICK['vf_coef'] * value_loss + entropy_bonus

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(bilateral_agent.parameters(), max_norm=0.5)
    optimizer.step()

    avg_loss = total_loss.item() / max(len(batch_states), 1)
    training_losses.append(avg_loss)
    
    elapsed = time.time() - start_time
    avg_return_last = np.mean(training_returns[-TRAIN_PARAMS_QUICK['num_steps']:])
    
    print(f" done. Return: {avg_return_last:.2f}, Loss: {avg_loss:.4f}, Elapsed: {elapsed:.1f}s")

print()
print(f"[OK] Training complete in {time.time() - start_time:.1f}s")
print(f"[INFO] Final episode return: {np.mean(training_returns[-TRAIN_PARAMS_QUICK['num_steps']:]):.4f}")
print("="*70 + "\n")
