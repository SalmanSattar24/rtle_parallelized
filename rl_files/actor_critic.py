import os
import random
from dataclasses import dataclass
import time
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
# sys path hacks
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from simulation.market_gym import Market 

@dataclass
class Args:
    # other options dirichlet, normal
    exp_name: str = 'log_normal'
    # exp_name: str = 'dirichlet'
    # exp_name: str = 'log_normal_learn_std'
    # exp_name: str = 'dirichlet'
    tag: Optional[str] = None
    """additional tag for the experiment, should be string type"""
    seed: int = 0
    """seed of the experiment"""
    eval_seed: int = 100
    """seed for evaluation"""
    bilateral: bool = False
    """if toggled, use bilateral market-making agent (BilateralAgentLogisticNormal)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model """
    evaluate: bool = True
    """whether to evaluate the model"""
    n_eval_episodes: int = int(1e4)
    n_eval_episodes: int = 5  # TEST: Quick evaluation (5 episodes)
    """the number of episodes to evaluate the model"""
    run_directory: str = 'runs'
    """directory for saving models"""
    run_name:  Optional[str] = None 
    """to be filled at runtime, should be string type"""

    # Algorithm specific arguments
    drop_feature: Optional[str] = 'drift'
    '''feature to be dropped from observation, can be "volume", "order_info", "drift", "None" or None'''
    env_type: str = "strategic"
    # noise, flow, strategic 
    """the id of the environment"""
    num_lots: int = 20
    num_lots: int = 40
    """the number of lots"""
    terminal_time: int = 150
    # terminal_time: int = 300
    """the terminal time for the execution agent"""
    time_delta: int = 15
    # this setting leads to 10 time steps. num of lots should be divisuble by 10
    """the time delta for the execution agent"""
    # total_timesteps: int = 200*128*100
    # total_timesteps = itertaions * n_cpus * n_steps (in each evnironment)
    # 10 iterations is one full episode 
    total_timesteps: int = 200*128*100
    total_timesteps: int = 20  # TEST: 5-min run (2 iterations)
    # debug
    # total_timesteps: int = 2*10
    # total_timesteps: int = 500*128*100
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    # num_envs: int = 128
    num_envs: int = 1  # TEST: Single environment
    # num_envs: int = 1
    # num_envs: int = 1
    """the number of parallel game environments"""
    # num_steps: int = 100
    num_steps: int = 10
    # num_steps: int = 10
    # less value bootstraping --> user more steps per environment
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    # do additional test here with gae_lambda = 0.95
    gae_lambda: float = 1.0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.5
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(config):
    def thunk():
        return Market(config)
    return thunk

class PinnedMemoryBuffer:
    """Helper class for efficient CPU↔GPU transfers using pinned memory"""
    def __init__(self, num_envs, obs_shape, device, enable_async=True):
        self.device = device
        self.enable_async = enable_async
        self.stream = torch.cuda.Stream(device) if enable_async and device.type == 'cuda' else None

        # Pre-allocate pinned CPU buffers
        self.obs_buffer = torch.zeros(
            num_envs, *obs_shape,
            dtype=torch.float32,
            pin_memory=(device.type == 'cuda')
        )
        self.reward_buffer = torch.zeros(
            num_envs,
            dtype=torch.float32,
            pin_memory=(device.type == 'cuda')
        )
        self.done_buffer = torch.zeros(
            num_envs,
            dtype=torch.float32,
            pin_memory=(device.type == 'cuda')
        )

    def transfer_to_device(self, obs_np, reward_np, done_np):
        """Transfer data from CPU (NumPy) to GPU with optional async"""
        # Copy numpy arrays to pinned memory
        self.obs_buffer.copy_(torch.from_numpy(obs_np), non_blocking=True)
        self.reward_buffer.copy_(torch.from_numpy(reward_np), non_blocking=True)
        self.done_buffer.copy_(torch.from_numpy(done_np), non_blocking=True)

        # Transfer to GPU
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                obs_gpu = self.obs_buffer.to(self.device, non_blocking=True)
                reward_gpu = self.reward_buffer.to(self.device, non_blocking=True)
                done_gpu = self.done_buffer.to(self.device, non_blocking=True)
        else:
            obs_gpu = self.obs_buffer.to(self.device, non_blocking=True)
            reward_gpu = self.reward_buffer.to(self.device, non_blocking=True)
            done_gpu = self.done_buffer.to(self.device, non_blocking=True)

        return obs_gpu, reward_gpu, done_gpu

    def synchronize(self):
        """Ensure all async transfers complete"""
        if self.stream is not None:
            self.stream.synchronize()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        n_hidden_units = 128 
        super().__init__()
        # critic network with 2 hidden layers
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )
        # action network with 2 hidden layers
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            # this is different than the logistic normal agent, no -1 here 
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)), std=1e-5),
        )
        # still use the same bias logic in the last layer [-1,-1, ... , -1, 1]
        x = -1.0*torch.ones(np.prod(envs.single_action_space.shape))
        x[-1] = 1.0
        self.actor_mean[-1].bias.data.copy_(x)
        # variance is scaled manually during training
        self.variance = 1.0 
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)        
        action_std = torch.ones_like(action_mean)*self.variance
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class AgentLogisticNormal(nn.Module):
    def __init__(self, envs, variance_scaling=True):
        n_hidden_units = 128 
        super().__init__()
        # critic network with 2 hidden layers
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )
        # action network with 2 hidden layers
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)-1), std=1e-5),
        )
        # custom bias in the last layer [-1,-1, ... , -1, 1]
        x = -1.0*torch.ones(np.prod(envs.single_action_space.shape)-1)
        x[-1] = 1.0
        self.actor_mean[-1].bias.data.copy_(x)

        # variance is scaled manually during training
        self.variance_scaling = variance_scaling
        if variance_scaling: 
            self.variance = 1.0 
        else:
            self.variance = None 
            self.log_std = nn.Parameter(torch.zeros(np.prod(envs.single_action_space.shape)-1), requires_grad=True)
 
        self.apply(layer_init)

    def check_parameters(self):
        """Returns True if any parameters in the model are NaN."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                return True, name
        return False, None

    def get_trunk_out(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)        
        if self.variance_scaling:
            action_std = torch.ones_like(action_mean)*self.variance
        else:   
            action_std = self.log_std.expand_as(action_mean)
            action_std = torch.exp(action_std)
        probs = Normal(action_mean, action_std)
        with torch.no_grad():
            if action is None:
                # sample base action, then apply logistic transformation, a = h(v)
                base_action = probs.sample()
                z = 1 + torch.sum(torch.exp(base_action), dim=1, keepdim=True)
                action = torch.exp(base_action)/z
                action = torch.cat((action, 1/z), dim=1)
            else:
                # use inverse logistic transform to get the base action v = h^{-1}(a)
                last_component = action[:,-1].reshape(-1,1)
                base_action = torch.log(action[:,:-1]/last_component)
        return action, probs.log_prob(base_action).sum(1), probs.entropy().sum(1), self.critic(x)

    def deterministic_action(self, x):
        # 
        # use mean base action, then apply logistic transformation, a = h(v)
        action_mean = self.actor_mean(x)        
        with torch.no_grad():
            # final_var = 0.32**2
            # base_action = action_mean + 0.5*final_var
            ## ADDING FINAL VARIANCE. SEE BELOW 
            base_action = action_mean
            z = 1 + torch.sum(torch.exp(base_action), dim=1, keepdim=True)
            action = torch.exp(base_action)/z
            action = torch.cat((action, 1/z), dim=1)
        return action
        

class DirichletAgent(nn.Module):
    def __init__(self, envs):         
        n_hidden_units = 128
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )    
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            # the last term scales the weights ! 
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)), std=1e-5),
        )

        # custom bias such that output after activation is [1,1,...,1,10]
        x = torch.log(torch.exp(torch.ones(np.prod(envs.single_action_space.shape)))-1)
        x[-1] = torch.log(torch.exp(torch.tensor(10.0))-1) 
        self.actor_mean[-1].bias.data.copy_(x)
        # print('Dirichlet agent initialized with bias ', self.actor_mean[-1].bias.data)
        # print(f'bias after softplus activation: {torch.nn.functional.softplus(self.actor_mean[-1].bias.data)}')
        # self.actor_var_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        # self.actor_var_scale = 1e-1
    
    def get_action_and_value(self, state, action=None): 
        mean = torch.nn.functional.softplus(self.actor_mean(state))  
        # print('Mean after softplus activation:', mean[0].detach().cpu().numpy())
        # scale = torch.nn.functional.softplus(self.actor_var_scale) + 1e-5
        # scale = 1e-5
        if torch.isnan(state).any():
            print("State contains NaN", state)
        if torch.isnan(mean).any():
            print("Mean contains Nan, state is ", mean)
        # if torch.isnan(scale).any():
        #     print("Scale contains NaNs:", scale)
        # scale =x 1e-5
        # concentrations = mean*scale 
        # concentrations = mean
        # concentrations = mean*sclae
        # probs = Dirichlet(mean)
        probs = Dirichlet(mean)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.variance, self.critic(state)

    def get_value(self, state):
        return self.critic(state)
    
    def deterministic_action(self, state):
        # get concentration parameters from the network
        mean = torch.nn.functional.softplus(self.actor_mean(state))
        return mean / torch.sum(mean, dim=1, keepdim=True)


class BilateralAgentLogisticNormal(nn.Module):
    """
    BILATERAL MARKET-MAKING AGENT (Phase 2.3)

    Dual-head policy network for simultaneous bid/ask order placement.
    - Shared trunk (critic) for state encoding
    - Two independent policy heads: actor_mean_bid and actor_mean_ask
    - Factored simplex policy: π(bid, ask | s) = π_bid(bid | s) * π_ask(ask | s)
    - Each action independently normalized to simplex

    Architecture:
        Input (observation) → Shared Trunk (128-128) → [Bid Head] → bid_action (K+1)
                                                    → [Ask Head] → ask_action (K+1)

    Action sampling:
        1. Sample bid_base ~ N(μ_bid, σ_I) independently
        2. Sample ask_base ~ N(μ_ask, σ_I) independently
        3. Apply logistic transform to each: a = softmax([exp(x), 1])
        4. Return (bid_action, ask_action) as tuple

    Loss factorization:
        log π = log π_bid + log π_ask
        entropy = entropy_bid + entropy_ask
    """

    def __init__(self, envs, variance_scaling=True):
        n_hidden_units = 128
        super().__init__()

        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape) - 1  # -1 because last component is computed from others

        # SHARED TRUNK: Common feature extraction with LayerNorm for stability
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_shape, n_hidden_units)),
            nn.LayerNorm(n_hidden_units),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.LayerNorm(n_hidden_units),
            nn.Tanh(),
        )

        # CRITIC HEAD: Value function (unchanged from unilateral)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )

        # BID POLICY HEAD: μ_bid for bid-side orders
        self.actor_mean_bid = nn.Sequential(
            layer_init(nn.Linear(n_hidden_units, action_dim), std=1e-5),
        )
        # Custom bias: [-1, -1, ..., -1, 1] → initial action skews toward inactive orders
        x_bid = -1.0 * torch.ones(action_dim)
        x_bid[-1] = 1.0
        self.actor_mean_bid[-1].bias.data.copy_(x_bid)

        # ASK POLICY HEAD: μ_ask for ask-side orders
        self.actor_mean_ask = nn.Sequential(
            layer_init(nn.Linear(n_hidden_units, action_dim), std=1e-5),
        )
        # Custom bias: same initialization
        x_ask = -1.0 * torch.ones(action_dim)
        x_ask[-1] = 1.0
        self.actor_mean_ask[-1].bias.data.copy_(x_ask)

        # VARIANCE: Shared across both heads
        self.variance_scaling = variance_scaling
        if variance_scaling:
            self.variance = 1.0
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        
        self.apply(layer_init)

    def check_parameters(self):
        """Returns True if any parameters in the model are NaN."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                return True, name
        return False, None

    def get_value(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        trunk_out = self.trunk(x)
        return self.critic(trunk_out)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Sample bilateral actions with factored policy.

        Args:
            x: observation tensor (batch_size, obs_dim)
            action: Tuple of (bid_action, ask_action) for computing log probs [optional]
            deterministic: If True, use means without sampling
        """
        # Phase G Sanitization: Ensure input is never NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        trunk_out = self.trunk(x)
        """
            actions: Tuple (bid_action, ask_action), each shape (batch_size, K+1)
            log_prob: Sum of bid and ask log probs, shape (batch_size,)
            entropy: Sum of bid and ask entropies, shape (batch_size,)
            value: State value, shape (batch_size, 1)
        """
        trunk_out = self.trunk(x)

        # Get policy means for both bid and ask
        bid_mean = self.actor_mean_bid(trunk_out)
        ask_mean = self.actor_mean_ask(trunk_out)

        # Construct standard deviations
        if self.variance_scaling:
            bid_std = torch.ones_like(bid_mean) * self.variance
            ask_std = torch.ones_like(ask_mean) * self.variance
        else:
            bid_std = torch.exp(self.log_std).expand_as(bid_mean)
            ask_std = torch.exp(self.log_std).expand_as(ask_mean)

        # Create distributions for bid and ask (independent)
        # Stability: Ensure scale is strictly positive and non-zero
        bid_std = torch.clamp(bid_std, min=1e-4)
        ask_std = torch.clamp(ask_std, min=1e-4)
        
        # Phase G Sanitization: Ensure means are never NaN/Inf before distribution creation
        bid_mean = torch.nan_to_num(bid_mean, nan=0.0)
        ask_mean = torch.nan_to_num(ask_mean, nan=0.0)
        
        bid_dist = Normal(bid_mean, bid_std)
        ask_dist = Normal(ask_mean, ask_std)

        # Sample or extract base actions
        with torch.no_grad():
            if action is None:
                # Sample phase: generate new actions
                bid_base = bid_dist.sample()
                ask_base = ask_dist.sample()

                # Apply logistic transform to each independently
                # bid_action: from K-dim base_action to K+1-dim simplex
                z_bid = 1 + torch.sum(torch.exp(bid_base), dim=1, keepdim=True)
                bid_action = torch.exp(bid_base) / z_bid
                bid_action = torch.cat((bid_action, 1 / z_bid), dim=1)

                # ask_action: from K-dim base_action to K+1-dim simplex
                z_ask = 1 + torch.sum(torch.exp(ask_base), dim=1, keepdim=True)
                ask_action = torch.exp(ask_base) / z_ask
                ask_action = torch.cat((ask_action, 1 / z_ask), dim=1)

            else:
                # Inference phase: invert simplex actions to get base actions
                bid_action, ask_action = action

                # Inverse logistic transform for bid
                bid_last = bid_action[:, -1].reshape(-1, 1)
                bid_base = torch.log(bid_action[:, :-1] / bid_last)

                # Inverse logistic transform for ask
                ask_last = ask_action[:, -1].reshape(-1, 1)
                ask_base = torch.log(ask_action[:, :-1] / ask_last)

        # Compute log probabilities (factored)
        bid_log_prob = bid_dist.log_prob(bid_base).sum(1)
        ask_log_prob = ask_dist.log_prob(ask_base).sum(1)
        log_prob_joint = bid_log_prob + ask_log_prob  # Factorization: π = π_bid * π_ask

        # Compute entropies (factored)
        bid_entropy = bid_dist.entropy().sum(1)
        ask_entropy = ask_dist.entropy().sum(1)
        entropy_joint = bid_entropy + ask_entropy

        # Pack actions as tuple for bilateral mode
        actions = (bid_action, ask_action)

        return actions, log_prob_joint, entropy_joint, self.critic(trunk_out)

    def deterministic_action(self, x):
        """
        Deterministic action selection using means (no noise).

        Returns:
            Tuple of (bid_action, ask_action) using mean policy
        """
        trunk_out = self.trunk(x)
        bid_mean = self.actor_mean_bid(trunk_out)
        ask_mean = self.actor_mean_ask(trunk_out)

        with torch.no_grad():
            # Logistic transform on bid
            z_bid = 1 + torch.sum(torch.exp(bid_mean), dim=1, keepdim=True)
            bid_action = torch.exp(bid_mean) / z_bid
            bid_action = torch.cat((bid_action, 1 / z_bid), dim=1)

            # Logistic transform on ask
            z_ask = 1 + torch.sum(torch.exp(ask_mean), dim=1, keepdim=True)
            ask_action = torch.exp(ask_mean) / z_ask
            ask_action = torch.cat((ask_action, 1 / z_ask), dim=1)

        return (bid_action, ask_action)


if __name__ == "__main__":
    """
    RTLE Actor-Critic Training with GPU Optimization

    PHASE 1 OPTIMIZATIONS (Implemented):
    - Pinned memory buffers for CPU↔GPU transfers (PinnedMemoryBuffer class)
    - Non-blocking async transfers with optional GPU streams
    - Reduces data transfer latency by ~50% (15-25% overall speedup expected)
    - Applies to both training and evaluation loops

    PHASE 2 OPTIMIZATIONS (Can be added):
    - CPU parallelization for observation generation with ProcessPoolExecutor
    - Parallel agent updates across environments
    - See actor_critic_phase2_utils.py for implementation details
    """
    args = tyro.cli(Args)
    print('starting the training process')
    print(f'environment set up: volume={args.num_lots}, market_env={args.env_type}')
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print('\n-----')
    print(f'batch_size={args.batch_size}, minibatch_size={args.minibatch_size}, num_iterations={args.num_iterations}, learning_rate={args.learning_rate}, num_iterations={args.num_iterations}, num_envs={args.num_envs}, num_steps_per_env={args.num_steps}, n_evalutation_episodes={args.n_eval_episodes}')
    print('-----')
    print('\n')
    print(f'time_delta={args.time_delta}, terminal_time={args.terminal_time}, lots={args.num_lots}')
    print('-----')

    # information should include: env_type, num_lots, seed, num_iterations, batch_size, algo_name
    # algo_name should describe the name of the algorithm, like log normal, dirichlet, normal softmax 
    # note that we are always using the actor critic algorithm, so we do not need to mention this 
    # naming convention: 
    assert args.drop_feature in ['volume', 'order_info', 'drift', 'None', None], f'unknown drop_feature {args.drop_feature}'
    
    if args.drop_feature == 'None':
        args.drop_feature = None

    feature_info = f'_{args.drop_feature}' if args.drop_feature is not None else ''
    if args.tag:
        print(f'additional tag for the experiment: {args.tag}')
        run_name = f"{args.env_type}_{args.num_lots}_seed_{args.seed}_eval_seed_{args.eval_seed}_eval_episodes_{args.n_eval_episodes}_num_iterations_{args.num_iterations}_bsize_{args.batch_size}_{args.exp_name}_{args.tag}{feature_info}"
        if args.tag == 'GAE':
            args.gae_lambda = 0.95
            print(f'using GAE with lambda = {args.gae_lambda}')
    else:
        print('no additional tag for the experiment')
        run_name = f"{args.env_type}_{args.num_lots}_seed_{args.seed}_eval_seed_{args.eval_seed}_eval_episodes_{args.n_eval_episodes}_num_iterations_{args.num_iterations}_bsize_{args.batch_size}_{args.exp_name}{feature_info}"    
    print(f'the run name is: {run_name}')
    args.run_name = run_name

    summary_path = f"{parent_dir}/tensorboard_logs/{run_name}"
    print(f'writing summary to {summary_path}:')
    writer = SummaryWriter(summary_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 0:
        # Choose the last GPU
        last_gpu = num_gpus - 1
        device = torch.device(f"cuda:{last_gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(last_gpu)} (cuda:{last_gpu})")
    else:
        # Fall back to CPU if no GPU is available
        device = torch.device("cpu")
        print("No GPU available, using CPU.")

    # TODO: additional option in config to remove features
    # This will then have an effect on how observations are generated  
    # include as option in config and add to name 
    # remove_featuer: "drift", "orders_info", "book_volumes"
    # then add this to the saving name 

    # environment setup
    print(f'dropping feature: {args.drop_feature}') if args.drop_feature is not None else print('not dropping any feature')
    configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 'seed': args.seed+s, 
                'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
    if args.exp_name == 'normal':
        # if we use just normal distribution, let the environment transform actions from R^n to the simplex 
        configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 'seed': args.seed+s, 
                    'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'transform_action': True, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
    env_fns = [make_env(config) for config in configs]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    observation, info = envs.reset(seed=args.seed)
    print(f'observation space: {envs.single_observation_space}, action space: {envs.single_action_space}')

    # PHASE 1: Setup pinned memory buffer for efficient data transfer
    pinned_buffer = PinnedMemoryBuffer(
        num_envs=args.num_envs,
        obs_shape=envs.single_observation_space.shape,
        device=device,
        enable_async=True
    )

    # agent set up. we have three cases log_normal, dirichlet, and normal
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
    elif args.exp_name == 'dirichlet':
        agent = DirichletAgent(envs).to(device)
    elif args.exp_name == 'normal':
        agent = Agent(envs).to(device)
    else:
        raise ValueError(f"unknown agent type: {args.exp_name}")
    print(f'the agent type is: {args.exp_name}')
    # print(f'the agent is: {agent}')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
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

    # start the simulation
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_gpu = torch.from_numpy(next_obs).float().to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.num_iterations < 2: 
        raise ValueError('num_iterations should be greater than 1')

    for iteration in range(0, args.num_iterations):
        print(f'iteration={iteration}')
        # Annealing the rate if instructed to do so.
        returns = []
        times = []
        drifts = []
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            print(f' the lerning rate is {lrnow}')        
        # manual standard deviation scalig. updated this to 0.1
        if args.exp_name == 'log_normal' or args.exp_name == 'normal' or args.exp_name == 'bilateral_log_normal':
            agent.variance = (0.32-1)*(iteration)/(args.num_iterations-1) + 1        
        # dirichlet agent does not use variance scaling 
        # agent.variance = 1 - iteration/(args.num_iterations+1) + 5e-1
        # keep same variance throughout the training
        # agent.variance = 1.0
        # if args.exp_name == 'normal' or args.exp_name == 'log_normal':            
            # print(f'the current variance is {agent.variance}')
        
        # this is the data collection loop
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs_gpu
            dones[step] = next_done

            # ALGO LOGIC: action logic
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
            next_done_np = np.logical_or(terminations, truncations)

            # Non-blocking async transfer with pinned memory
            next_obs_gpu, reward_gpu, next_done_gpu = pinned_buffer.transfer_to_device(
                next_obs_np,
                reward_np,
                next_done_np.astype(np.float32)
            )
            # Ensure transfers complete before using tensors
            pinned_buffer.synchronize()

            rewards[step] = reward_gpu.view(-1)
            next_done = next_done_gpu

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        returns.append(info['reward'])
                        times.append(info['time'])
                        drifts.append(info['drift'])
        
        writer.add_scalar("charts/return", np.mean(returns), global_step)
        writer.add_scalar("charts/time", np.mean(times), global_step)
        writer.add_scalar("charts/drift", np.mean(drifts), global_step)        
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_gpu).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # could remove gamma and lambda if this is 1 anyways 
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # shuffle indices could be removed since we are doing only one epoch
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # log probs are computed with old actions
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

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss modified 
                pg_loss = -mb_advantages * newlogprob
                pg_loss = pg_loss.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Compute entropy bonus (encourage exploration)
                entropy_loss = entropy.mean()
                # Total loss: policy loss + value loss - entropy bonus (negative to maximize entropy)
                loss = pg_loss + v_loss * args.vf_coef - entropy_loss * args.ent_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Log variance only for agents that have variance scaling
        if hasattr(agent, 'variance') and (args.exp_name == 'log_normal' or args.exp_name == 'normal' or args.exp_name == 'bilateral_log_normal'):
            writer.add_scalar("values/variance", agent.variance, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"{parent_dir}/models/{run_name}.pt"        
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    
    if args.evaluate:
        # deterministic_actions
        print('\n starting evaluation')
        print('using deterministic actions for evaluation')

        # Use SINGLE non-vectorized environment to avoid AsyncVectorEnv final_info issues
        config = {'market_env': args.env_type, 'execution_agent': 'rl_agent', 'volume': args.num_lots,
                  'seed': args.eval_seed, 'terminal_time': args.terminal_time, 'time_delta': args.time_delta,
                  'drop_feature': args.drop_feature}
        print('evaluation config:')
        print(config)

        # Create single Market environment directly (not vectorized)
        from simulation.market_gym import Market
        env_eval = Market(config)
        print('evaluation environment is created')

        obs, _ = env_eval.reset()
        obs_gpu = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # Add batch dim for agent
        episodic_returns = []
        start_time = time.time()
        max_eval_steps = args.n_eval_episodes * 1000  # Safety timeout
        eval_step = 0

        print(f'Running {args.n_eval_episodes} evaluation episodes (deterministic actions)...')
        while len(episodic_returns) < args.n_eval_episodes and eval_step < max_eval_steps:
            with torch.no_grad():
                # always use deterministic action for evaluation
                actions = agent.deterministic_action(obs_gpu)  # Input: (1, obs_dim), Output: (1, action_dim) or tuple

                if args.bilateral:
                    # actions is tuple (bid_action, ask_action)
                    bid_action, ask_action = actions
                    env_actions = (bid_action.squeeze(0).cpu().numpy(), ask_action.squeeze(0).cpu().numpy())
                else:
                    env_actions = actions.squeeze(0).cpu().numpy()

                next_obs_np, _, terminated, truncated, info = env_eval.step(env_actions)

            obs_gpu = torch.from_numpy(next_obs_np).float().to(device).unsqueeze(0)  # Add batch dim back
            eval_step += 1

            # DEBUG: Print step info
            if eval_step % 100 == 0:
                print(f"  Step {eval_step}: Episodes={len(episodic_returns)}, "
                      f"Terminated={terminated}, "
                      f"Remaining volume={info.get('volume', 'N/A')}")

            # For non-vectorized env, check terminated directly
            if terminated:
                reward = info['reward']
                print(f"  Episode {len(episodic_returns)+1} completed! Reward={reward:.4f}, Volume={info.get('volume', 'N/A')}")
                episodic_returns.append(reward)
                obs, _ = env_eval.reset()
                obs_gpu = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # Add batch dim for next episode

        env_eval.close()
        print(f'evaluation time: {time.time()-start_time:.1f}s')
        print(f'reward length: {len(episodic_returns)}/{args.n_eval_episodes}')

        # Warn if we hit the step limit before collecting all episodes
        if len(episodic_returns) < args.n_eval_episodes:
            print(f"WARNING: Evaluation hit max_eval_steps limit ({eval_step}/{max_eval_steps}) before collecting all episodes!")
            print(f"Only collected {len(episodic_returns)}/{args.n_eval_episodes} episodes.")

        rewards = np.array(episodic_returns) if len(episodic_returns) > 0 else np.array([])
        if len(rewards) == 0:
            print("ERROR: No episodes completed during evaluation!")
        else:
            assert args.run_name is not None, "run_name should be set"
            # use tags such as long_horizon, GAE, or whatever
            file_name = f'{parent_dir}/rewards/{args.run_name}.npz'
            print(f'save rewards to {file_name}')
            np.savez(file_name, rewards=rewards)
        # if args.tag is not None:
        #     file_name = f'{parent_dir}/rewards/{args.run_name}_{args.tag}.npz'
        # else:

    envs.close()
    writer.close()



