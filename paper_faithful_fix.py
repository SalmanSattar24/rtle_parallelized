"""
Phase C: Paper-Faithful Simplification
Strips out custom penalties, increases trajectory count, uses paper's reward/hyperparams.
"""
import nbformat
import os

path = 'c:/All-Code/CSCI-566/rtle_parallelized/bilateral_mm_agent.ipynb'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ============================================================
# 1. CELL 17: Config — Start with noise, I_max=10
# ============================================================
nb.cells[17].source = """# Paper-Faithful Configuration (Cheridito & Weiss 2026)
# Start with NOISE environment to validate pipeline, then move to flow/strategic

TRAIN_CONFIG = {
    'market_env': 'noise',       # Start simple: noise-only market
    'execution_agent': 'rl_agent',
    'volume': 10,                # Paper uses 10 lots with I_max=10
    'seed': 42,
    'terminal_time': 150,        # Paper: 150 seconds
    'time_delta': 15,            # Paper: 15s intervals → 10 time steps
    'drop_feature': 'drift',     # Paper default
    'inventory_max': 10,         # Paper: tight cap experiment
    'penalty_weight': 0.0,       # NO quadratic penalty (paper doesn't use one)
}

EVAL_CONFIG = {
    'market_env': 'noise',
    'execution_agent': 'rl_agent',
    'volume': 10,
    'seed': 100,
    'terminal_time': 150,
    'time_delta': 15,
    'drop_feature': 'drift',
    'inventory_max': 10,
    'penalty_weight': 0.0,
}

EVAL_EPISODES = 1000

print("[OK] Paper-Faithful Config loaded (noise, I_max=10, volume=10, T=150, dt=15)")
"""

# ============================================================
# 2. CELL 21: Replace project_action_risk with paper's Q_max quota
# ============================================================
nb.cells[21].source = """import torch

def project_action_quota(a, current_inv, side="bid", inventory_max=10, q_max_base=10):
    \"\"\"
    Paper-faithful quota projection.
    Q_max = min(q_max_base, I_max - |I(t)|) per side per step.
    
    This mechanically prevents inventory overflow through the action space,
    no artificial penalties needed.
    \"\"\"
    x = torch.clamp(a, min=1e-8)
    
    # Paper: Q_max = min(q_max_base, I_max - |I(t)|)
    if side == "bid":
        room = max(0.0, inventory_max - current_inv)  # Room to buy more
    else:  # ask
        room = max(0.0, inventory_max + current_inv)  # Room to sell more
    
    q_max = min(q_max_base, room)
    
    if q_max <= 0:
        # No room: force everything to hold (last bucket)
        result = torch.zeros_like(x)
        result[..., -1] = 1.0
        return result
    
    # Scale active mass (non-hold buckets) proportional to quota
    volume = 10  # Must match TRAIN_CONFIG['volume']
    active_mass_limit = q_max / volume
    
    active_mass = torch.sum(x[..., :-1], dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(active_mass), active_mass_limit / (active_mass + 1e-8))
    
    x[..., :-1] *= scale
    # Renormalize to ensure valid simplex
    total = torch.sum(x, dim=-1, keepdim=True)
    x = x / (total + 1e-8)
    
    return x

print("[OK] Paper-faithful quota projection loaded (Q_max = min(10, I_max - |I(t)|))")
"""

# ============================================================
# 3. CELL 22: Paper-faithful training loop
# ============================================================
nb.cells[22].source = r'''print("=" * 70)
print("STEP 7: TRAIN BILATERAL AGENT (Paper-Faithful PPO)")
print("=" * 70)
print()

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

# === Paper Hyperparameters (Cheridito & Weiss 2026) ===
NUM_TRAIN_ITERS = 400       # Paper: 400 gradient steps
EPISODES_PER_ITER = 64      # Paper: 1280 trajectories; we use 64 as compute-feasible approximation
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256

BASE_LR = 5e-4              # Paper: 0.0005
CLIP_EPS = 0.20
VF_COEF = 0.5               # Paper standard
MAX_GRAD_NORM = 0.5         # Paper: clip gradients at 0.5

GAMMA_TRAIN = 1.0           # Paper: undiscounted (gamma=1)
GAE_LAMBDA = 1.0            # Paper: full returns (lambda=1)

# Paper: variance schedule sigma_init=1.0 -> sigma_final=0.1
ENT_COEF_START = 0.25
ENT_COEF_END = 0.05

# Fresh start
bilateral_agent = BilateralAgentLogisticNormal(market).to(device)
optimizer = torch.optim.Adam(bilateral_agent.parameters(), lr=BASE_LR, eps=1e-5)
training_returns = []
training_losses = []
start_time = time.time()

best_val_score = -float('inf')
best_val_return = -float('inf')
best_state_dict = copy.deepcopy(bilateral_agent.state_dict())

def quick_eval(agent, episodes=120, seed_base=20000):
    vals = []
    for i in range(episodes):
        cfg = dict(EVAL_CONFIG)
        cfg['seed'] = seed_base + i
        m_raw = Market(cfg)
        m = EnvWrapper(m_raw)
        obs, _ = m.reset()
        ep_ret = 0.0
        current_inventory = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                bid_action, ask_action = agent.deterministic_action(obs_tensor)
                bid_action = project_action_quota(bid_action, current_inventory, side="bid", inventory_max=EVAL_CONFIG['inventory_max'])
                ask_action = project_action_quota(ask_action, current_inventory, side="ask", inventory_max=EVAL_CONFIG['inventory_max'])
                env_action = (bid_action[0].cpu().numpy(), ask_action[0].cpu().numpy())
            obs, reward, terminated, truncated, info = m.step(env_action)
            ep_ret += float(reward)
            current_inventory = info.get("net_inventory", 0)
            if terminated or truncated:
                break
        vals.append(ep_ret)
    vals = np.array(vals)
    cvar5 = np.mean(np.sort(vals)[:max(1, len(vals)//20)])
    outlier = np.mean(vals < -200.0)
    return float(np.mean(vals)), float(np.std(vals)), float(cvar5), float(outlier)

for iteration in range(NUM_TRAIN_ITERS):
    iter_start = time.time()

    # Entropy/variance schedule (linear decay)
    frac = iteration / max(NUM_TRAIN_ITERS - 1, 1)
    ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * frac

    # Variance schedule (paper: sigma from 1.0 to 0.1)
    if hasattr(bilateral_agent, 'set_variance'):
        new_var = 1.0 - 0.9 * frac  # 1.0 -> 0.1
        bilateral_agent.set_variance(new_var)

    obs_buf, bid_buf, ask_buf, old_logprob_buf, adv_buf, ret_buf = [], [], [], [], [], []

    for episode in range(EPISODES_PER_ITER):
        cfg = dict(TRAIN_CONFIG)
        cfg['seed'] = iteration * 10000 + episode
        m_raw = Market(cfg)
        market_wrap = EnvWrapper(m_raw)
        obs, _ = market_wrap.reset()
        current_inventory = 0

        ep_obs, ep_bid, ep_ask, ep_logprobs, ep_values, ep_rewards = [], [], [], [], [], []
        ep_return = 0.0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            bid_action, ask_action, log_prob, value = bilateral_agent.get_action_and_value(obs_tensor)

            # Paper-faithful quota projection
            bid_action = project_action_quota(bid_action, current_inventory, side="bid", inventory_max=TRAIN_CONFIG['inventory_max'])
            ask_action = project_action_quota(ask_action, current_inventory, side="ask", inventory_max=TRAIN_CONFIG['inventory_max'])

            # Recompute log prob after projection
            _, log_prob_proj, _, _ = bilateral_agent.get_action_and_value(obs_tensor, action=(bid_action, ask_action))

            env_action = (bid_action[0].detach().cpu().numpy(), ask_action[0].detach().cpu().numpy())
            obs_next, reward, terminated, truncated, info = market_wrap.step(env_action)

            # Update inventory
            current_inventory = info.get("net_inventory", 0)

            # Paper-faithful reward: just use the raw reward from the environment
            reward = float(reward)

            ep_obs.append(obs_tensor.squeeze(0).detach())
            ep_bid.append(bid_action.squeeze(0).detach())
            ep_ask.append(ask_action.squeeze(0).detach())
            ep_logprobs.append(log_prob_proj.squeeze().detach())
            ep_values.append(value.squeeze().detach())
            ep_rewards.append(reward)

            ep_return += reward
            obs = obs_next

            if terminated or truncated:
                break

        rewards_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device)
        values_t = torch.stack(ep_values)
        adv_t = torch.zeros_like(rewards_t)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(ep_rewards))):
            delta = rewards_t[t] + GAMMA_TRAIN * next_value - values_t[t]
            gae = delta + GAMMA_TRAIN * GAE_LAMBDA * gae
            adv_t[t] = gae
            next_value = values_t[t]

        ret_t = adv_t + values_t

        obs_buf.extend(ep_obs)
        bid_buf.extend(ep_bid)
        ask_buf.extend(ep_ask)
        old_logprob_buf.extend(ep_logprobs)
        adv_buf.extend([a.detach() for a in adv_t])
        ret_buf.extend([r.detach() for r in ret_t])

        training_returns.append(ep_return)

    b_obs = torch.stack(obs_buf)
    b_bid = torch.stack(bid_buf)
    b_ask = torch.stack(ask_buf)
    b_old_logprobs = torch.stack(old_logprob_buf)
    b_advantages = torch.stack(adv_buf)
    b_returns = torch.stack(ret_buf)

    # Standard advantage normalisation
    adv_mean = b_advantages.mean()
    adv_std = b_advantages.std()
    if torch.isnan(adv_std) or adv_std < 1e-6:
        b_advantages = b_advantages - adv_mean
    else:
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

    batch_size = b_obs.shape[0]
    mb_size = min(MINIBATCH_SIZE, batch_size)
    inds = np.arange(batch_size)

    iter_losses = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(inds)
        for start in range(0, batch_size, mb_size):
            mb_inds = inds[start:start + mb_size]
            mb_inds_t = torch.tensor(mb_inds, dtype=torch.long, device=device)

            mb_obs = b_obs[mb_inds_t]
            mb_actions = (b_bid[mb_inds_t], b_ask[mb_inds_t])
            mb_old_logprobs = b_old_logprobs[mb_inds_t]
            mb_adv = b_advantages[mb_inds_t]
            mb_returns = b_returns[mb_inds_t]

            _, new_logprob, entropy, new_value = bilateral_agent.get_action_and_value(mb_obs, action=mb_actions)
            new_value = new_value.squeeze()

            # Clean PPO loss
            log_ratio = new_logprob - mb_old_logprobs
            ratio = torch.exp(log_ratio)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.smooth_l1_loss(new_value, mb_returns)
            entropy_bonus = entropy.mean()

            total_loss = actor_loss + VF_COEF * value_loss - ent_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bilateral_agent.parameters(), max_norm=MAX_GRAD_NORM)

            if torch.isnan(total_loss) or torch.isinf(total_loss) or torch.isnan(grad_norm):
                continue

            optimizer.step()
            iter_losses.append(float(total_loss.item()))

    mean_loss = float(np.mean(iter_losses)) if len(iter_losses) > 0 else float('nan')
    training_losses.append(mean_loss)

    if (iteration + 1) % 20 == 0:
        val_mean, val_std, val_cvar5, val_outlier = quick_eval(bilateral_agent, episodes=120, seed_base=30000 + iteration * 100)
        val_score = val_mean - 0.20 * val_std - 100.0 * val_outlier
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_return = val_mean
            best_state_dict = copy.deepcopy(bilateral_agent.state_dict())
            tag = " [BEST]"
        else:
            tag = ""
    else:
        val_mean, val_std, val_cvar5, val_outlier, val_score = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        tag = ""

    if (iteration + 1) % 20 == 0 or iteration < 5:
        elapsed_iter = time.time() - iter_start
        elapsed_total = time.time() - start_time
        avg_return_20 = np.mean(training_returns[-20:]) if len(training_returns) >= 20 else np.mean(training_returns)
        current_var = bilateral_agent.variance if hasattr(bilateral_agent, 'variance') and bilateral_agent.variance is not None else float('nan')
        print(
            f"[{iteration+1:3d}/{NUM_TRAIN_ITERS}] {elapsed_iter:5.1f}s | "
            f"Total: {elapsed_total:6.1f}s | Avg20: {avg_return_20:8.2f} | "
            f"Loss: {mean_loss:8.4f} | Ent: {ent_coef:6.4f} | Var: {current_var:5.3f} | "
            f"Val120: {val_mean:8.2f}\u00b1{val_std:6.2f} | CVaR5: {val_cvar5:8.2f} | Out<-200: {val_outlier:6.3f} | "
            f"Score: {val_score:8.2f}{tag}"
        )

bilateral_agent.load_state_dict(best_state_dict)

print(f"\n[OK] Training complete in {time.time() - start_time:.1f}s")
print(f"[INFO] Training return (last 20): {np.mean(training_returns[-20:]):.4f}")
print(f"[INFO] Training loss (last 20): {np.mean(training_losses[-20:]):.4f}")
print(f"[INFO] Best validation mean return (120 eps): {best_val_return:.4f}")
print(f"[INFO] Best validation score: {best_val_score:.4f}")
print("=" * 70 + "\n")
'''

with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("PAPER_FAITHFUL_FIX_COMPLETE")
