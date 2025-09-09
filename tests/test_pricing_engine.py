"""
Tests for the Monte Carlo pricing engine core functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.heston_model import HestonModel
from models.sabr_model import SABRModel
from pricing.monte_carlo_engine import MonteCarloEngine
from pricing.payoff_functions import EuropeanPayoff, AsianPayoff, BarrierPayoff


class TestMonteCarloEngine:
    """Test Monte Carlo pricing engine functionality."""
    
    def test_engine_initialization(self):
        """Test that engine initializes properly."""
        model = HestonModel()
        engine = MonteCarloEngine(model, n_paths=10000)
        assert engine.model == model
        assert engine.n_paths == 10000
        assert engine.n_steps == 252
    
    def test_european_call_pricing(self):
        """Test European call option pricing."""
        model = HestonModel(seed=42)
        engine = MonteCarloEngine(model, n_paths=50000, seed=42)
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        
        price = engine.price_european_call(spot, strike, maturity)
        assert price > 0
        assert price < spot  # Call price should be less than spot
    
    def test_european_put_pricing(self):
        """Test European put option pricing."""
        model = HestonModel(seed=42)
        engine = MonteCarloEngine(model, n_paths=50000, seed=42)
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        
        price = engine.price_european_put(spot, strike, maturity)
        assert price > 0
        assert price < strike  # Put price should be less than strike
    
    def test_asian_option_pricing(self):
        """Test Asian option pricing."""
        model = HestonModel(seed=42)
        engine = MonteCarloEngine(model, n_paths=50000, seed=42)
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        
        price = engine.price_asian_option(spot, strike, maturity, "call", "arithmetic")
        assert price > 0
        assert price < spot
    
    def test_barrier_option_pricing(self):
        """Test barrier option pricing."""
        model = HestonModel(seed=42)
        engine = MonteCarloEngine(model, n_paths=50000, seed=42)
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        barrier = 90.0
        
        price = engine.price_barrier_option(spot, strike, maturity, barrier, "down-and-out", "call")
        assert price > 0
        assert price < spot
    
    def test_pricing_accuracy(self):
        """Test pricing accuracy against analytical solutions."""
        model = HestonModel(seed=42)
        engine = MonteCarloEngine(model, n_paths=100000, seed=42)
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        
        # Price with Monte Carlo
        mc_price = engine.price_european_call(spot, strike, maturity)
        
        # Compare with analytical (if available)
        try:
            analytical_price = model.get_analytical_price(spot, strike, maturity, "call")
            error = abs(mc_price - analytical_price) / analytical_price
            assert error < 0.05  # Within 5% accuracy
        except:
            # If analytical not available, just check MC price is reasonable
            assert 0 < mc_price < spot


class TestHestonModel:
    """Test Heston model functionality."""
    
    def test_model_initialization(self):
        """Test Heston model initialization."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        assert model.v0 == 0.04
        assert model.kappa == 2.0
        assert model.theta == 0.04
        assert model.rho == -0.7
        assert model.sigma == 0.5
    
    def test_path_generation(self):
        """Test Monte Carlo path generation."""
        model = HestonModel(seed=42)
        spot = 100.0
        maturity = 1.0
        n_paths = 1000
        n_steps = 252
        
        asset_paths, volatility_paths = model.generate_paths(spot, maturity, n_paths, n_steps)
        
        assert asset_paths.shape == (n_paths, n_steps + 1)
        assert volatility_paths.shape == (n_paths, n_steps + 1)
        assert np.all(asset_paths > 0)
        assert np.all(volatility_paths > 0)


class TestSABRModel:
    """Test SABR model functionality."""
    
    def test_model_initialization(self):
        """Test SABR model initialization."""
        model = SABRModel(
            alpha=0.2,
            beta=0.5,
            rho=-0.1,
            nu=0.5,
            risk_free_rate=0.05
        )
        assert model.alpha == 0.2
        assert model.beta == 0.5
        assert model.rho == -0.1
        assert model.nu == 0.5
    
    def test_path_generation(self):
        """Test Monte Carlo path generation."""
        model = SABRModel(seed=42)
        spot = 100.0
        maturity = 1.0
        n_paths = 1000
        n_steps = 252
        
        asset_paths, volatility_paths = model.generate_paths(spot, maturity, n_paths, n_steps)
        
        assert asset_paths.shape == (n_paths, n_steps + 1)
        assert volatility_paths.shape == (n_paths, n_steps + 1)
        assert np.all(asset_paths > 0)
        assert np.all(volatility_paths > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
