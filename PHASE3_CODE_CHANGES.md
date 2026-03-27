# Phase 3 - Exact Code Modifications

This document shows the exact modifications made to `rl_files/actor_critic.py` for Phase 3.

## Modification 1: Configuration Flag

**Location**: `@dataclass class Args` section (around line 24)

```python
# ADDED (lines 37-38):
bilateral: bool = False
"""if toggled, use bilateral market-making agent (BilateralAgentLogisticNormal)"""
```

**Usage**: `--bilateral` flag when running training

---

## Modification 2: Agent Instantiation

**Location**: After environment setup, before optimizer creation (around line 636)

```python
# BEFORE:
if args.exp_name == 'log_normal':
    agent = AgentLogisticNormal(envs).to(device)
elif args.exp_name == 'log_normal_learn_std':
    agent = AgentLogisticNormal(envs, variance_scaling=False).to(device)
# ... other cases ...

# AFTER (lines 638-655):
# Also add bilateral variant for market-making
print(f'Bilateral mode: {args.bilateral}')

if args.bilateral:
    print('Using BilateralAgentLogisticNormal for bilateral market-making')
    agent = BilateralAgentLogisticNormal(envs).to(device)
    args.exp_name = 'bilateral_log_normal'
elif args.exp_name == 'log_normal':
    agent = AgentLogisticNormal(envs).to(device)
elif args.exp_name == 'log_normal_learn_std':
    agent = AgentLogisticNormal(envs, variance_scaling=False).to(device)
# ... other cases remain unchanged ...
```

---

## Modification 3: Action Storage Initialization

**Location**: ALGO Logic: Storage setup (around line 659)

```python
# BEFORE:
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# AFTER (lines 659-673):
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

# Handle bilateral vs unilateral action storage
if args.bilateral:
    # For bilateral: store bid and ask actions separately
    bid_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    ask_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
else:
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
```

---

## Modification 4: Action Sampling Loop

**Location**: Data collection loop, action logic (around line 695)

```python
# BEFORE:
with torch.no_grad():
    action, logprob, _, value = agent.get_action_and_value(next_obs_gpu)
    values[step] = value.flatten()
actions[step] = action
logprobs[step] = logprob

# TRY NOT TO MODIFY: execute the game and log data.
next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())

# AFTER (lines 713-732):
with torch.no_grad():
    action, logprob, _, value = agent.get_action_and_value(next_obs_gpu)
    values[step] = value.flatten()

if args.bilateral:
    # action is tuple (bid_action, ask_action)
    bid_action, ask_action = action
    bid_actions[step] = bid_action
    ask_actions[step] = ask_action
    # Pass tuple to environment
    env_action = (bid_action.cpu().numpy(), ask_action.cpu().numpy())
else:
    actions[step] = action
    env_action = action.cpu().numpy()

logprobs[step] = logprob

# TRY NOT TO MODIFY: execute the game and log data.
# PHASE 1 OPTIMIZATION: Use pinned memory for efficient transfers
next_obs_np, reward_np, terminations, truncations, infos = envs.step(env_action)
```

---

## Modification 5: Variance Schedule

**Location**: Variance scaling in main iteration loop (around line 697)

```python
# BEFORE:
if args.exp_name == 'log_normal' or args.exp_name == 'normal':
    agent.variance = (0.32-1)*(iteration)/(args.num_iterations-1) + 1

# AFTER:
if args.exp_name == 'log_normal' or args.exp_name == 'normal' or args.exp_name == 'bilateral_log_normal':
    agent.variance = (0.32-1)*(iteration)/(args.num_iterations-1) + 1
```

---

## Modification 6: Action Flattening for Training

**Location**: Flatten the batch section (around line 747)

```python
# BEFORE:
b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs = logprobs.reshape(-1)
b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values.reshape(-1)

# AFTER (lines 779-792):
b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs = logprobs.reshape(-1)

if args.bilateral:
    # For bilateral: reshape bid and ask actions separately
    b_bid_actions = bid_actions.reshape((-1,) + envs.single_action_space.shape)
    b_ask_actions = ask_actions.reshape((-1,) + envs.single_action_space.shape)
    # Combine into list of tuples for passing to agent
    b_actions = [(b_bid_actions[i], b_ask_actions[i]) for i in range(len(b_bid_actions))]
else:
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values.reshape(-1)
```

---

## Modification 7: Loss Computation with Minibatch Actions

**Location**: Optimizing the policy and value network, inside minibatch loop (around line 762)

```python
# BEFORE:
_, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
logratio = newlogprob - b_logprobs[mb_inds]
ratio = logratio.exp()

# AFTER (lines 803-813):
if args.bilateral:
    # Extract bid and ask actions for this minibatch
    mb_bid_actions = torch.stack([b_bid_actions[i] for i in mb_inds])
    mb_ask_actions = torch.stack([b_ask_actions[i] for i in mb_inds])
    mb_actions = (mb_bid_actions, mb_ask_actions)
else:
    mb_actions = b_actions[mb_inds]

_, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], mb_actions)
logratio = newlogprob - b_logprobs[mb_inds]
ratio = logratio.exp()
```

---

## Modification 8: Variance Logging

**Location**: Recording rewards for plotting (around line 859)

```python
# BEFORE:
if args.exp_name == 'log_normal' or args.exp_name == 'normal':
    writer.add_scalar("values/variance", agent.variance, global_step)

# AFTER:
# Log variance only for agents that have variance scaling
if hasattr(agent, 'variance') and (args.exp_name == 'log_normal' or args.exp_name == 'normal' or args.exp_name == 'bilateral_log_normal'):
    writer.add_scalar("values/variance", agent.variance, global_step)
```

---

## Modification 9: Evaluation Loop

**Location**: Evaluation episode step (around line 895)

```python
# BEFORE:
with torch.no_grad():
    # always use deterministic action for evaluation
    actions = agent.deterministic_action(obs_gpu)
    next_obs_np, _, terminated, truncated, info = env_eval.step(actions.squeeze(0).cpu().numpy())

# AFTER (lines 895-907):
with torch.no_grad():
    # always use deterministic action for evaluation
    actions = agent.deterministic_action(obs_gpu)

    if args.bilateral:
        # actions is tuple (bid_action, ask_action)
        bid_action, ask_action = actions
        env_actions = (bid_action.squeeze(0).cpu().numpy(), ask_action.squeeze(0).cpu().numpy())
    else:
        env_actions = actions.squeeze(0).cpu().numpy()

    next_obs_np, _, terminated, truncated, info = env_eval.step(env_actions)
```

---

## Summary Statistics

- **Total lines added**: ~50 lines
- **Number of modification points**: 9
- **Lines modified in existing code**: ~15
- **New variables introduced**: 2 (bid_actions, ask_actions when bilateral=True)
- **Backward compatibility**: 100% (all existing code paths preserved)
- **New imports**: None (uses existing classes only)
- **Test files created**: 1 (`tests/test_phase3_training.py`)

## Verification

All modifications have been:
- ✓ Syntax validated (file parses without errors)
- ✓ Logically reviewed (coherent with design)
- ✓ Compared with BilateralAgentLogisticNormal interface
- ✓ Confirmed backward compatible (bilateral=False uses existing paths)

---

**Phase 3 Implementation**: COMPLETE ✓
