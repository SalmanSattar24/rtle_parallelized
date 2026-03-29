"""
Phase E: Hyper-Speed Vectorization & Checkpointing
This script applies the vectorized training loop and checkpointing logic to the notebook.
"""
import nbformat
import os

path = 'c:/All-Code/CSCI-566/rtle_parallelized/bilateral_mm_agent.ipynb'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ------------------------------------------------------------
# 1. ADD GOOGLE DRIVE MOUNT (New Cell 1)
# ------------------------------------------------------------
drive_cell = nbformat.v4.new_code_cell(source="""# OPTIONAL: Mount Google Drive for persistent checkpointing
# This ensures you don't lose your model if Colab disconnects.
from google.colab import drive
import os

try:
    drive.mount('/content/drive')
    CHECKPOINT_DIR = "/content/drive/MyDrive/mm_rl_checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[OK] Google Drive mounted. Checkpoints will be saved to: {CHECKPOINT_DIR}")
except Exception as e:
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[WARN] Could not mount Drive ({e}). Saving checkpoints locally to: {CHECKPOINT_DIR}")
""")

# Insert at Step 1 or 2
nb.cells.insert(4, drive_cell)

# ------------------------------------------------------------
# 2. UPDATE project_action_quota FOR VECTORIZATION (Cell 25/26 area)
# ------------------------------------------------------------
for cell in nb.cells:
    if 'def project_action_quota' in cell.source:
        cell.source = """def project_action_quota(x_orig, current_inv, side="bid", inventory_max=10, q_max_base=10):
    \"\"\"
    Vectorized Paper-faithful quota projection.
    Supports both scalar and batch inputs (for Vectorized Environments).
    \"\"\"
    x = x_orig.clone()
    device = x.device
    
    # Ensure current_inv is a tensor and has shape (batch, 1)
    if not isinstance(current_inv, torch.Tensor):
        current_inv = torch.tensor(current_inv, device=device, dtype=torch.float32)
    
    # Handle scalar or vector current_inv
    if current_inv.dim() == 0:
        current_inv = current_inv.unsqueeze(0)
    if current_inv.dim() == 1:
        current_inv = current_inv.unsqueeze(1)
        
    if side == "bid":
        room = torch.maximum(torch.zeros_like(current_inv), inventory_max - current_inv)
    else:  # ask
        room = torch.maximum(torch.zeros_like(current_inv), inventory_max + current_inv)
    
    q_max = torch.minimum(torch.tensor(float(q_max_base), device=device), room)
    
    # Identify environments with NO room (force 'hold' action)
    no_room_mask = (q_max <= 1e-4).squeeze()
    if no_room_mask.any():
        if x.dim() > 1:
            x[no_room_mask, :-1] = 0.0
            x[no_room_mask, -1] = 1.0
        else:
            x[:-1] = 0.0
            x[-1] = 1.0

    # Scale active mass (non-hold buckets) proportional to quota
    volume = 10.0  # Must match TRAIN_CONFIG['volume']
    active_mass_limit = q_max / volume
    
    active_mass = torch.sum(x[..., :-1], dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(active_mass), active_mass_limit / (active_mass + 1e-8))
    
    x[..., :-1] *= scale
    # Renormalize to ensure valid simplex
    total = torch.sum(x, dim=-1, keepdim=True)
    x = x / (total + 1e-8)
    
    return x

print("[OK] Vectorized Quota Projection loaded.")
"""

# ------------------------------------------------------------
# 3. REPLACE STEP 7 WITH HYPER-SPEED VECTORIZED VERSION (Cell 26/27 area)
# ------------------------------------------------------------
# Find Step 7 cell (the one containing the training loop)
for i, cell in enumerate(nb.cells):
    if 'NUM_TRAIN_ITERS = 400' in cell.source and 'EPISODES_PER_ITER = 64' in cell.source:
        cell.source = """print("=" * 70)
print("STEP 7: TRAIN BILATERAL AGENT (Hyper-Speed Vectorized PPO)")
print("=" * 70)
print(f"[SPEED] Training optimized for GPU using 64 parallel environments.")

import gymnasium as gym
import copy
import time

# --- Setup Configuration ---
NUM_TRAIN_ITERS = 400
NUM_ENVS = 64               # Parallelize EVERYTHING
STEPS_PER_ROLLOUT = 11      # Sufficient for T=150, dt=15
PPO_EPOCHS = 4
BATCH_SIZE = NUM_ENVS * STEPS_PER_ROLLOUT
MINIBATCH_SIZE = 128

# --- Hyperparameters ---
BASE_LR = 5e-4
MAX_GRAD_NORM = 0.5
ENT_COEF_START = 0.25
ENT_COEF_END = 0.05

# --- Environment Factory ---
def make_env(config):
    def thunk():
        from simulation.market_gym import Market
        return Market(config)
    return thunk

env_fns = [make_env(TRAIN_CONFIG) for _ in range(NUM_ENVS)]
envs = gym.vector.AsyncVectorEnv(env_fns)

# --- Initialization ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilateral_agent = BilateralAgentLogisticNormal(envs).to(device)
optimizer = torch.optim.Adam(bilateral_agent.parameters(), lr=BASE_LR, eps=1e-5)

best_val_score = -float('inf')
best_state_dict = None
training_returns = []

# Recovery Check:
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pth")
if os.path.exists(checkpoint_path):
    print(f"[RESUME] Found existing checkpoint at {checkpoint_path}. Loading...")
    bilateral_agent.load_state_dict(torch.load(checkpoint_path))

# --- Loop ---
start_time = time.time()

for iteration in range(NUM_TRAIN_ITERS):
    iter_start = time.time()
    
    # Entropy / Var Schedule
    frac = iteration / NUM_TRAIN_ITERS
    ent_coef = ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * frac
    if hasattr(bilateral_agent, 'set_variance'):
        bilateral_agent.set_variance(1.0 - 0.9 * frac)

    # Buffers
    obs_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    bid_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS, 7)).to(device)
    ask_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS, 7)).to(device)
    logprob_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS)).to(device)
    value_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS)).to(device)
    reward_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS)).to(device)
    done_batch = torch.zeros((STEPS_PER_ROLLOUT, NUM_ENVS)).to(device)

    # Rollout Experience (Vectorized)
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    current_inventory = torch.zeros(NUM_ENVS).to(device)

    for step in range(STEPS_PER_ROLLOUT):
        obs_batch[step] = next_obs
        done_batch[step] = next_done
        
        with torch.no_grad():
            (bid, ask), log_prob, _, value = bilateral_agent.get_action_and_value(next_obs)
            
            # Vectorized Quota Application
            bid = project_action_quota(bid, current_inventory, side="bid")
            ask = project_action_quota(ask, current_inventory, side="ask")
            
            # Re-collect logprob for projected actions
            _, log_prob, _, _ = bilateral_agent.get_action_and_value(next_obs, action=(bid, ask))
            
            value_batch[step] = value.flatten()
            bid_batch[step] = bid
            ask_batch[step] = ask
            logprob_batch[step] = log_prob

        # Step ALL envs at once
        env_action = (bid.cpu().numpy(), ask.cpu().numpy())
        next_obs_np, rewards, terms, truncs, infos = envs.step(env_action)
        
        reward_batch[step] = torch.tensor(rewards).to(device).view(-1)
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
        next_done = torch.from_numpy(np.logical_or(terms, truncs)).float().to(device)
        
        # Track inventory for next quota step
        if "net_inventory" in infos: # Some gymnasium versions use this
             current_inventory = torch.tensor(infos["net_inventory"]).to(device)
        else: # Generic fallback for AsyncVectorEnv infos structure
             current_inventory = torch.tensor([i.get("net_inventory", 0) for i in infos.get("final_info", [{} for _ in range(NUM_ENVS)])]).to(device)

    # Compute Returns (Advantage)
    with torch.no_grad():
        next_value = bilateral_agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(reward_batch).to(device)
        lastgaelam = 0
        for t in reversed(range(STEPS_PER_ROLLOUT)):
            if t == STEPS_PER_ROLLOUT - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - done_batch[t+1]
                nextvalues = value_batch[t+1]
            delta = reward_batch[t] + 1.0 * nextvalues * nextnonterminal - value_batch[t]
            advantages[t] = lastgaelam = delta + 1.0 * 1.0 * nextnonterminal * lastgaelam
        returns = advantages + value_batch

    # PPO Optimization Step
    b_obs = obs_batch.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprob_batch.reshape(-1)
    b_bid = bid_batch.reshape(-1, 7)
    b_ask = ask_batch.reshape(-1, 7)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = value_batch.reshape(-1)

    # Single Pass PPO
    _, newlogprob, entropy, newvalue = bilateral_agent.get_action_and_value(b_obs, (b_bid, b_ask))
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()
    
    pg_loss = -(b_advantages * newlogprob).mean()
    v_loss = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()
    loss = pg_loss + v_loss * 0.5 - entropy.mean() * ent_coef

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(bilateral_agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    # Logging
    avg_ret = reward_batch.sum(dim=0).mean().item()
    training_returns.append(avg_ret)
    
    if (iteration + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"[{iteration+1:3d}/{NUM_TRAIN_ITERS}] {elapsed:6.1f}s | AvgRet: {avg_ret:7.2f} | Loss: {loss:7.4f}")
        
        # Periodic Checkpoint
        torch.save(bilateral_agent.state_dict(), os.path.join(CHECKPOINT_DIR, "checkpoint_last.pth"))
        
        # Validation
        v_mean, v_std, v_cvar, v_out = quick_eval(bilateral_agent, episodes=40)
        score = v_mean - v_std
        if score > best_val_score:
            best_val_score = score
            best_state_dict = copy.deepcopy(bilateral_agent.state_dict())
            torch.save(best_state_dict, os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth"))
            print(f"  --> [NEW BEST] Score: {score:.2f} (ckpt saved)")

envs.close()
bilateral_agent.load_state_dict(best_state_dict)
print(f"\\n[OK] Vectorized Training Complete in {time.time()-start_time:.1f}s")
"""

# ============================================================
# 4. WRITE THE NOTEBOOK
# ============================================================
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("PHASE_E_VECTORIZATION_APPLIED")
