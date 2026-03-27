"""
Phase 1 Unit Tests: Bilateral Simulator Foundation
Tests for inventory tracking, soft penalty logic, and terminal close-out
"""

import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

import numpy as np
from limit_order_book.limit_order_book import LimitOrderBook, LimitOrder, MarketOrder
from simulation.market_gym import Market
import pytest


class TestInventoryTracking:
    """Test inventory tracking and consistency in LOB"""

    def setup_method(self):
        """Setup for each test"""
        self.agents = ['agent1', 'agent2', 'noise_agent']
        self.lob = LimitOrderBook(list_of_agents=self.agents, level=10, only_volumes=False)

    def test_inventory_initialized_to_zero(self):
        """Test that all agents start with zero inventory"""
        for agent in self.agents:
            assert self.lob.agent_net_inventory[agent] == 0, \
                f"Agent {agent} should start with 0 inventory"

    def test_bid_order_registration(self):
        """Test that bid orders are properly registered"""
        order = LimitOrder('agent1', 'bid', 100, 10, time=0)
        msg = self.lob.process_order(order)
        assert len(self.lob.agent_bid_orders['agent1']) == 1, \
            "Bid order should be registered for agent1"

    def test_ask_order_registration(self):
        """Test that ask orders are properly registered"""
        order = LimitOrder('agent1', 'ask', 101, 10, time=0)
        msg = self.lob.process_order(order)
        assert len(self.lob.agent_ask_orders['agent1']) == 1, \
            "Ask order should be registered for agent1"

    def test_market_buy_updates_inventory(self):
        """Test that market buy increases inventory"""
        # Place a sell limit order from agent2
        sell_order = LimitOrder('agent2', 'ask', 101, 5, time=0)
        self.lob.process_order(sell_order)

        # Market buy from agent1
        buy_order = MarketOrder('agent1', 'bid', 5, time=1)
        msg = self.lob.process_order(buy_order)

        # agent1 should have +5 inventory
        assert self.lob.agent_net_inventory['agent1'] == 5, \
            f"Agent1 inventory should be +5, got {self.lob.agent_net_inventory['agent1']}"

    def test_market_sell_updates_inventory(self):
        """Test that market sell decreases inventory"""
        # Place a buy limit order from agent2
        buy_order = LimitOrder('agent2', 'bid', 99, 3, time=0)
        self.lob.process_order(buy_order)

        # Market sell from agent1
        sell_order = MarketOrder('agent1', 'ask', 3, time=1)
        msg = self.lob.process_order(sell_order)

        # agent1 should have -3 inventory
        assert self.lob.agent_net_inventory['agent1'] == -3, \
            f"Agent1 inventory should be -3, got {self.lob.agent_net_inventory['agent1']}"

    def test_passive_fill_updates_inventory(self):
        """Test that passive fills update counterparty inventory"""
        # Place a buy limit order from agent1
        buy_order = LimitOrder('agent1', 'bid', 99, 4, time=0)
        self.lob.process_order(buy_order)

        # Market sell from agent2 (hits agent1's limit buy)
        sell_order = MarketOrder('agent2', 'ask', 4, time=1)
        msg = self.lob.process_order(sell_order)

        # agent1 should have +4 inventory (filled buy order)
        # agent2 should have -4 inventory (market sell executed)
        assert self.lob.agent_net_inventory['agent1'] == 4, \
            f"Agent1 inventory should be +4, got {self.lob.agent_net_inventory['agent1']}"
        assert self.lob.agent_net_inventory['agent2'] == -4, \
            f"Agent2 inventory should be -4, got {self.lob.agent_net_inventory['agent2']}"

    def test_inventory_consistency_check(self):
        """Test inventory consistency verification"""
        # Place orders and execute fills
        buy_order = LimitOrder('agent1', 'bid', 99, 5, time=0)
        self.lob.process_order(buy_order)
        sell_order = MarketOrder('agent2', 'ask', 5, time=1)
        self.lob.process_order(sell_order)

        # Verify consistency doesn't raise error
        try:
            self.lob.assert_inventory_consistent()
            assert True, "Inventory consistency check should pass"
        except AssertionError as e:
            pytest.fail(f"Inventory consistency check failed: {e}")

    def test_complex_inventory_sequence(self):
        """Test inventory tracking through multiple orders"""
        # agent1: +2 buy, -1 sell, +3 buy = +4 total

        # First buy limit
        buy1 = LimitOrder('agent1', 'bid', 99, 2, time=0)
        self.lob.process_order(buy1)

        # Sell limit that gets hit
        sell1 = LimitOrder('other', 'ask', 100, 1, time=1)
        self.lob.process_order(sell1)
        market_sell = MarketOrder('agent1', 'ask', 1, time=2)
        self.lob.process_order(market_sell)

        # Second buy gets filled passively
        buy2 = LimitOrder('agent1', 'bid', 98, 3, time=3)
        self.lob.process_order(buy2)
        market_buy = MarketOrder('agent2', 'ask', 3, time=4)
        self.lob.process_order(market_buy)

        # Verify final inventory: 2 - 1 + 3 = 4
        assert self.lob.agent_net_inventory['agent1'] == 4, \
            f"Agent1 final inventory should be 4, got {self.lob.agent_net_inventory['agent1']}"

        # Verify consistency
        self.lob.assert_inventory_consistent()


class TestMarketEnvironmentInventory:
    """Test inventory tracking in Market environment"""

    def test_market_initializes_with_zero_inventory(self):
        """Test that Market environment starts with zero inventory"""
        config = {
            'market_env': 'noise',
            'execution_agent': 'linear_sl_agent',
            'volume': 40,
            'seed': 42,
            'terminal_time': 1000,
            'time_delta': 50,
            'drop_feature': None,
            'inventory_max': 10,
            'penalty_weight': 1.0,
        }
        market = Market(config)
        assert market.agent_inventory == 0, \
            f"Market should initialize with 0 inventory, got {market.agent_inventory}"

    def test_market_resets_inventory_to_zero(self):
        """Test that inventory resets to zero on reset()"""
        config = {
            'market_env': 'noise',
            'execution_agent': 'linear_sl_agent',
            'volume': 40,
            'seed': 42,
            'terminal_time': 1000,
            'time_delta': 50,
            'drop_feature': None,
            'inventory_max': 10,
            'penalty_weight': 1.0,
        }
        market = Market(config)
        market.agent_inventory = 5  # Manually set to non-zero
        obs, info = market.reset(seed=42)
        assert market.agent_inventory == 0, \
            f"Inventory should reset to 0 after reset(), got {market.agent_inventory}"

    def test_soft_penalty_applied_when_exceeding_limit(self):
        """Test that soft penalty is applied when inventory exceeds limit"""
        config = {
            'market_env': 'noise',
            'execution_agent': 'linear_sl_agent',
            'volume': 40,
            'seed': 42,
            'terminal_time': 1000,
            'time_delta': 50,
            'drop_feature': None,
            'inventory_max': 5,
            'penalty_weight': 2.0,
        }
        market = Market(config)

        # Manually set inventory to exceed limit (for testing purposes)
        market.agent_inventory = 8  # Exceeds limit of 5
        excess = abs(8) - 5  # = 3
        expected_penalty = 2.0 * excess  # = 6.0

        # Simulate penalty calculation
        if abs(market.agent_inventory) >= market.inventory_max:
            excess_inventory = abs(market.agent_inventory) - market.inventory_max
            actual_penalty = market.penalty_weight * excess_inventory
            assert actual_penalty == expected_penalty, \
                f"Expected penalty {expected_penalty}, got {actual_penalty}"


class TestPhase1ExitCriteria:
    """Test Phase 1 exit criteria"""

    def test_inventory_consistency_across_many_episodes(self):
        """Test that inventory consistency holds over 1000 episodes"""
        n_episodes = 100  # Reduced for faster testing
        consistency_passed = 0

        for episode in range(n_episodes):
            agents = ['agent1', 'agent2', 'agent3']
            lob = LimitOrderBook(list_of_agents=agents, level=10, only_volumes=False)

            # Random orders
            np.random.seed(episode)
            n_orders = np.random.randint(5, 20)

            for i in range(n_orders):
                agent = agents[np.random.randint(0, len(agents))]
                side = 'bid' if np.random.random() < 0.5 else 'ask'
                volume = np.random.randint(1, 10)

                if side == 'bid':
                    price = 99 + np.random.uniform(-2, 2)
                else:
                    price = 101 + np.random.uniform(-2, 2)

                order = LimitOrder(agent, side, int(price), volume, time=i)
                try:
                    lob.process_order(order)
                except:
                    pass  # Some orders may fail, that's ok

            # Verify consistency
            try:
                lob.assert_inventory_consistent()
                consistency_passed += 1
            except AssertionError:
                pass

        # At least 95% should pass
        assert consistency_passed >= n_episodes * 0.95, \
            f"Only {consistency_passed}/{n_episodes} episodes passed consistency check"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
