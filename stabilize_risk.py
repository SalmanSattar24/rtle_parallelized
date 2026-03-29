import nbformat
import torch
import copy

path = r'c:\All-Code\CSCI-566\rtle_parallelized\bilateral_mm_agent.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Update Configurations (Volume: 20, Max 20 lots, 10 lot room to 30)
train_cfg_code = '''# SOTA: Phase 6 Configuration (Cheridito & Weiss 2026 Alignment)
TRAIN_CONFIG = {
    'market_env': 'strategic', # Noise + Tactical + Strategic Traders
    'execution_agent': 'rl_agent',
    'volume': 20, # Reduced to 20 for 10-lot buffer to 30 limit
    'seed': 42,
    'terminal_time': 500,
    'time_delta': 50,
    'drop_feature': None,
    'inventory_max': 30, 
    'penalty_weight': 0.005,
}

TRAIN_PARAMS = {
    'num_iterations': 200,
    'num_steps': 10,
    'batch_size': 10,
    'learning_rate': 5e-4,
    'entropy_coef': 0.05,
    'vf_coef': 0.5,
    'gamma': 1.0,
    'gae_lambda': 1.0,
}

EVAL_CONFIG = {
    'market_env': 'strategic',
    'execution_agent': 'rl_agent',
    'volume': 20, # Reduced to 20
    'seed': 100,
    'terminal_time': 500,
    'time_delta': 50,
    'drop_feature': None,
    'inventory_max': 30,
    'penalty_weight': 0.005,
}

EVAL_EPISODES = 1000

print("[OK] Phase 6 Configuration loaded (Volume: 20, Limit: 30)")
'''

# 2. Update project_action_risk (Unified Version)
unified_risk_code = '''import torch

def project_action_risk(a, current_inv, side="bid", inventory_max=30.0):
    """
    Phase 6 Hard Quota Rule: Ensures orders never exceed remaining inventory room.
    """
    x = torch.clamp(a, min=1e-8)
    
    # 1. Calculate room relative to death-limit (30)
    # current_inv = (net_inventory)
    if side == "bid":
        # Bid = Buying. Room to +30.
        room = max(0.0, inventory_max - current_inv)
    else:
        # Ask = Selling. Room to -30.
        room = max(0.0, inventory_max + current_inv)
        
    # 2. Dynamic Order Cap: Limit per-step impact to 5 lots (25% of 20 total)
    order_cap = min(room, 5.0)
    active_mass_limit = order_cap / 20.0
    
    # 3. Apply Limit to Level 0 (Market) and Levels 1-5 (Limit)
    active_mass = torch.sum(x[..., :-1], dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(active_mass), active_mass_limit / (active_mass + 1e-8))
    
    x[..., :-1] *= scale
    
    # 4. Final Re-normalization: All remaining mass -> Inactive (index 6)
    total_scaled = torch.sum(x, dim=-1, keepdim=True)
    x[..., -1] += (1.0 - total_scaled).clamp(min=0)
    
    return x / (x.sum(dim=-1, keepdim=True) + 1e-8)
'''

# 3. Step 7 Loop Improvements
# We need to increase SAFETY_BONUS and make sure current_inventory is passed
safety_bonus_boost = '''            # Safety bonus: reward for staying close to baseline
            bid_dist = torch.sum(torch.abs(bid_action - baseline_prior)).item()
            ask_dist = torch.sum(torch.abs(ask_action - baseline_prior)).item()
            safety_bonus = 0.5 / (1.0 + bid_dist + ask_dist) # Increased to 0.5
            reward = float(reward) + safety_bonus
'''

# Replace cells
for i, cell in enumerate(nb.cells):
    if 'TRAIN_CONFIG =' in cell.source:
        cell.source = train_cfg_code
    if 'def project_action_risk' in cell.source:
        cell.source = unified_risk_code
    if 'safety_bonus =' in cell.source and 'iteration' in nb.cells[i-1].source: # Loop cell
        # Update bonus in loop
        cell.source = cell.source.replace('safety_bonus = SAFETY_BONUS', 'safety_bonus = 0.5')

# Clear duplicate project_action_risk in Step 8 (Eval Cell)
for i, cell in enumerate(nb.cells):
    if 'Step 8: Evaluate' in cell.source or 'Step 9: Evaluate' in cell.source:
        if 'def project_action_risk' in cell.source:
             cell.source = cell.source.replace('def project_action_risk', '# def project_action_risk (REMOVED: USING UNIFIED)', 1)

with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Stabilization complete.")
