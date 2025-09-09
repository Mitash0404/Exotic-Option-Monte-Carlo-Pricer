"""
Payoff functions for various option types.
"""

import numpy as np
from abc import ABC, abstractmethod


class PayoffFunction(ABC):
    """
    Abstract base class for option payoff functions.
    """
    
    def __init__(self, strike: float, option_type: str = "call"):
        """
        Initialize the payoff function.
        """
        self.strike = strike
        self.option_type = option_type.lower()
        
        if self.option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
    
    @abstractmethod
    def calculate_payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculate option payoffs for given asset prices.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the payoff function."""
        return f"{self.__class__.__name__}(strike={self.strike}, type={self.option_type})"


class EuropeanPayoff(PayoffFunction):
    """
    European option payoff function.
    """
    
    def calculate_payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculate European option payoffs.
        """
        if asset_prices.ndim == 1:
            # Single path
            if self.option_type == "call":
                return np.maximum(asset_prices - self.strike, 0)
            else:  # put
                return np.maximum(self.strike - asset_prices, 0)
        else:
            # Multiple paths
            if self.option_type == "call":
                return np.maximum(asset_prices - self.strike, 0)
            else:  # put
                return np.maximum(self.strike - asset_prices, 0)


class AsianPayoff(PayoffFunction):
    """
    Asian option payoff function.
    """
    
    def __init__(self, strike: float, option_type: str = "call", averaging_type: str = "arithmetic"):
        """
        Initialize Asian option payoff function.
        """
        super().__init__(strike, option_type)
        self.averaging_type = averaging_type.lower()
        
        if self.averaging_type not in ["arithmetic", "geometric"]:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")
    
    def calculate_payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculate Asian option payoffs.
        """
        if asset_prices.ndim != 2:
            raise ValueError("Asian payoff requires 2D asset price array")
        
        # Calculate average price along each path
        if self.averaging_type == "arithmetic":
            average_prices = np.mean(asset_prices, axis=1)
        else:  # geometric
            average_prices = np.exp(np.mean(np.log(asset_prices), axis=1))
        
        # Calculate payoffs
        if self.option_type == "call":
            return np.maximum(average_prices - self.strike, 0)
        else:  # put
            return np.maximum(self.strike - average_prices, 0)


class BarrierPayoff(PayoffFunction):
    """
    Barrier option payoff function.
    """
    
    def __init__(self, 
                 strike: float, 
                 barrier: float,
                 barrier_type: str = "down-and-out",
                 option_type: str = "call"):
        """
        Initialize barrier option payoff function.
        """
        super().__init__(strike, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()
        
        valid_barrier_types = ["up-and-out", "down-and-out", "up-and-in", "down-and-in"]
        if self.barrier_type not in valid_barrier_types:
            raise ValueError(f"barrier_type must be one of {valid_barrier_types}")
    
    def calculate_payoff(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Calculate barrier option payoffs.
        """
        if asset_prices.ndim != 2:
            raise ValueError("Barrier payoff requires 2D asset price array")
        
        n_paths = asset_prices.shape[0]
        payoffs = np.zeros(n_paths)
        
        # Check barrier conditions for each path
        for i in range(n_paths):
            path_prices = asset_prices[i, :]
            
            # Determine if barrier is hit
            barrier_hit = self._check_barrier_condition(path_prices)
            
            # Calculate payoff based on barrier type
            if self._is_option_active(barrier_hit):
                # Calculate vanilla option payoff at maturity
                final_price = path_prices[-1]
                if self.option_type == "call":
                    payoffs[i] = max(final_price - self.strike, 0)
                else:  # put
                    payoffs[i] = max(self.strike - final_price, 0)
            else:
                payoffs[i] = 0.0
        
        return payoffs
    
    def _check_barrier_condition(self, path_prices: np.ndarray) -> bool:
        """
        Check if barrier condition is met for a given path.
        """
        if self.barrier_type in ["up-and-out", "up-and-in"]:
            # Check if upper barrier is hit
            return np.any(path_prices >= self.barrier)
        else:  # down-and-out, down-and-in
            # Check if lower barrier is hit
            return np.any(path_prices <= self.barrier)
    
    def _is_option_active(self, barrier_hit: bool) -> bool:
        """
        Determine if the option is active based on barrier condition.
        """
        if self.barrier_type in ["up-and-out", "down-and-out"]:
            # Out options are active if barrier is NOT hit
            return not barrier_hit
        else:  # up-and-in, down-and-in
            # In options are active if barrier IS hit
            return barrier_hit

