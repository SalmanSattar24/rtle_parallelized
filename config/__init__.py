"""
Configuration module

This package contains configuration dictionaries for all agents and the simulation environment.
"""

# Import all configuration dictionaries
from .config import (
    noise_agent_config,
    sl_agent_config,
    linear_sl_agent_config,
    market_agent_config,
    rl_agent_config,
    strategic_agent_config,
    initial_agent_config,
    observation_agent_config,
)

__all__ = [
    "noise_agent_config",
    "sl_agent_config",
    "linear_sl_agent_config",
    "market_agent_config",
    "rl_agent_config",
    "strategic_agent_config",
    "initial_agent_config",
    "observation_agent_config",
]
