"""
Monte Carlo pricing engine for exotic options.
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from ..models.stochastic_volatility import StochasticVolatilityModel


class MonteCarloEngine:
    """
    High-performance Monte Carlo pricing engine for exotic options.
    """
    
    def __init__(self, 
                 model: StochasticVolatilityModel,
                 n_paths: int = 100000,
                 n_steps: int = 252,
                 seed: Optional[int] = None):
        """
        Initialize the Monte Carlo engine.
        """
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        
        # Performance tracking
        self.pricing_times = []
    
    def price_european_call(self, 
                           spot: float,
                           strike: float,
                           maturity: float,
                           risk_free_rate: Optional[float] = None) -> float:
        """
        Price a European call option using Monte Carlo simulation.
        """
        start_time = time.time()
        
        # Use provided risk-free rate or model default
        if risk_free_rate is not None:
            r = risk_free_rate
        else:
            r = self.model.risk_free_rate
        
        # Generate Monte Carlo paths
        asset_paths, _ = self.model.generate_paths(
            spot=spot,
            maturity=maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed
        )
        
        # Calculate payoffs at maturity
        final_prices = asset_paths[:, -1]
        payoffs = np.maximum(final_prices - strike, 0)
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-r * maturity)
        
        # Performance tracking
        pricing_time = time.time() - start_time
        self.pricing_times.append(pricing_time)
        
        return option_price
    
    def price_european_put(self, 
                          spot: float,
                          strike: float,
                          maturity: float,
                          risk_free_rate: Optional[float] = None) -> float:
        """
        Price a European put option using Monte Carlo simulation.
        """
        start_time = time.time()
        
        # Use provided risk-free rate or model default
        if risk_free_rate is not None:
            r = risk_free_rate
        else:
            r = self.model.risk_free_rate
        
        # Generate Monte Carlo paths
        asset_paths, _ = self.model.generate_paths(
            spot=spot,
            maturity=maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed
        )
        
        # Calculate payoffs at maturity
        final_prices = asset_paths[:, -1]
        payoffs = np.maximum(strike - final_prices, 0)
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-r * maturity)
        
        # Performance tracking
        pricing_time = time.time() - start_time
        self.pricing_times.append(pricing_time)
        
        return option_price
    
    def price_asian_option(self, 
                          spot: float,
                          strike: float,
                          maturity: float,
                          option_type: str = "call",
                          averaging_type: str = "arithmetic",
                          risk_free_rate: Optional[float] = None) -> float:
        """
        Price an Asian option using Monte Carlo simulation.
        """
        start_time = time.time()
        
        # Use provided risk-free rate or model default
        if risk_free_rate is not None:
            r = risk_free_rate
        else:
            r = self.model.risk_free_rate
        
        # Generate Monte Carlo paths
        asset_paths, _ = self.model.generate_paths(
            spot=spot,
            maturity=maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed
        )
        
        # Calculate Asian payoffs
        from .payoff_functions import AsianPayoff
        payoff_func = AsianPayoff(strike, option_type, averaging_type)
        payoffs = payoff_func.calculate_payoff(asset_paths)
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-r * maturity)
        
        # Performance tracking
        pricing_time = time.time() - start_time
        self.pricing_times.append(pricing_time)
        
        return option_price
    
    def price_barrier_option(self, 
                           spot: float,
                           strike: float,
                           maturity: float,
                           barrier: float,
                           barrier_type: str = "down-and-out",
                           option_type: str = "call",
                           risk_free_rate: Optional[float] = None) -> float:
        """
        Price a barrier option using Monte Carlo simulation.
        """
        start_time = time.time()
        
        # Use provided risk-free rate or model default
        if risk_free_rate is not None:
            r = risk_free_rate
        else:
            r = self.model.risk_free_rate
        
        # Generate Monte Carlo paths
        asset_paths, _ = self.model.generate_paths(
            spot=spot,
            maturity=maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            seed=self.seed
        )
        
        # Calculate barrier payoffs
        from .payoff_functions import BarrierPayoff
        payoff_func = BarrierPayoff(strike, barrier, barrier_type, option_type)
        payoffs = payoff_func.calculate_payoff(asset_paths)
        
        # Calculate option price
        option_price = np.mean(payoffs) * np.exp(-r * maturity)
        
        # Performance tracking
        pricing_time = time.time() - start_time
        self.pricing_times.append(pricing_time)
        
        return option_price
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        """
        if not self.pricing_times:
            return {}
        
        return {
            'total_pricings': len(self.pricing_times),
            'average_pricing_time': np.mean(self.pricing_times),
            'total_time': np.sum(self.pricing_times),
            'fastest_pricing': np.min(self.pricing_times),
            'slowest_pricing': np.max(self.pricing_times)
        }
    
    def __str__(self) -> str:
        """String representation of the Monte Carlo engine."""
        return (f"MonteCarloEngine(model={self.model.__class__.__name__}, "
                f"n_paths={self.n_paths}, n_steps={self.n_steps})")
