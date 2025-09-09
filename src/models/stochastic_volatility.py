"""
Base class for stochastic volatility models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class StochasticVolatilityModel(ABC):
    """
    Abstract base class for stochastic volatility models.
    
    This class defines the interface that all stochastic volatility models
    must implement for use with the Monte Carlo pricing engine.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the stochastic volatility model.
        
        Args:
            risk_free_rate: Risk-free interest rate (default: 0.05)
        """
        self.risk_free_rate = risk_free_rate
        
    @abstractmethod
    def generate_paths(self, 
                      spot: float, 
                      maturity: float, 
                      n_paths: int, 
                      n_steps: int,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo paths for the underlying asset and volatility.
        
        Args:
            spot: Initial spot price
            maturity: Time to maturity in years
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (asset_paths, volatility_paths) as numpy arrays
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """
        Set model parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate that model parameters are within reasonable bounds.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # This is a base implementation - subclasses should override
        return True
    
    def __str__(self) -> str:
        """String representation of the model."""
        params = self.get_parameters()
        param_str = ", ".join([f"{k}={v:.4f}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()
