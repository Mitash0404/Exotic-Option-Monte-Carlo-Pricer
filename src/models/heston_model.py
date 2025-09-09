"""
Heston stochastic volatility model implementation.
"""

import numpy as np
from typing import Tuple, Optional
from .stochastic_volatility import StochasticVolatilityModel


class HestonModel(StochasticVolatilityModel):
    """
    Heston stochastic volatility model.
    """
    
    def __init__(self, 
                 v0: float = 0.04,
                 kappa: float = 2.0,
                 theta: float = 0.04,
                 rho: float = -0.7,
                 sigma: float = 0.5,
                 risk_free_rate: float = 0.05,
                 seed: Optional[int] = None):
        """
        Initialize the Heston model.
        """
        super().__init__(risk_free_rate)
        
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.sigma = sigma
        self.seed = seed
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid Heston model parameters")
    
    def generate_paths(self, 
                      spot: float, 
                      maturity: float, 
                      n_paths: int, 
                      n_steps: int,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo paths using the Heston model.
        """
        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)
        
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        asset_paths = np.zeros((n_paths, n_steps + 1))
        volatility_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        asset_paths[:, 0] = spot
        volatility_paths[:, 0] = self.v0
        
        # Generate correlated random numbers
        z1 = np.random.standard_normal((n_paths, n_steps))
        z2 = np.random.standard_normal((n_paths, n_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
        
        # Generate paths
        for i in range(n_steps):
            # Current values
            S_t = asset_paths[:, i]
            V_t = volatility_paths[:, i]
            
            # Ensure variance is positive
            V_t = np.maximum(V_t, 1e-8)
            sqrt_V_t = np.sqrt(V_t)
            
            # Asset price evolution
            drift_S = (self.risk_free_rate - 0.5 * V_t) * dt
            diffusion_S = sqrt_V_t * sqrt_dt * z1[:, i]
            asset_paths[:, i + 1] = S_t * np.exp(drift_S + diffusion_S)
            
            # Variance evolution
            drift_V = self.kappa * (self.theta - V_t) * dt
            diffusion_V = self.sigma * sqrt_V_t * sqrt_dt * z2[:, i]
            volatility_paths[:, i + 1] = np.maximum(V_t + drift_V + diffusion_V, 1e-8)
        
        return asset_paths, volatility_paths
    
    def get_parameters(self) -> dict:
        """Get the current model parameters."""
        return {
            'v0': self.v0,
            'kappa': self.kappa,
            'theta': self.theta,
            'rho': self.rho,
            'sigma': self.sigma,
            'risk_free_rate': self.risk_free_rate
        }
    
    def set_parameters(self, **kwargs) -> None:
        """Set model parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def validate_parameters(self) -> bool:
        """
        Validate Heston model parameters.
        """
        # Check Feller condition: 2*kappa*theta > sigma^2
        if 2 * self.kappa * self.theta <= self.sigma**2:
            return False
        
        # Check parameter bounds
        if (self.v0 <= 0 or self.kappa <= 0 or self.theta <= 0 or 
            self.sigma <= 0 or abs(self.rho) > 1):
            return False
        
        return True
