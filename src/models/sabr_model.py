"""
SABR stochastic volatility model implementation.
"""

import numpy as np
from typing import Tuple, Optional
from .stochastic_volatility import StochasticVolatilityModel


class SABRModel(StochasticVolatilityModel):
    """
    SABR (Stochastic Alpha Beta Rho) stochastic volatility model.
    
    The SABR model describes the evolution of an asset price F(t) and its
    stochastic volatility alpha(t) by the following system of SDEs:
    
    dF(t) = alpha(t) * F(t)^beta * dW1(t)
    dalpha(t) = nu * alpha(t) * dW2(t)
    
    where:
    - alpha is the initial volatility
    - beta is the CEV parameter (0 <= beta <= 1)
    - nu is the volatility of volatility
    - rho is the correlation between asset and volatility processes
    """
    
    def __init__(self, 
                 alpha: float = 0.2,
                 beta: float = 0.5,
                 rho: float = -0.1,
                 nu: float = 0.5,
                 risk_free_rate: float = 0.05):
        """
        Initialize the SABR model.
        
        Args:
            alpha: Initial volatility (default: 0.2)
            beta: CEV parameter (default: 0.5)
            rho: Correlation between asset and volatility (default: -0.1)
            nu: Volatility of volatility (default: 0.5)
            risk_free_rate: Risk-free interest rate (default: 0.05)
        """
        super().__init__(risk_free_rate)
        
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid SABR model parameters")
    
    def generate_paths(self,
                      spot: float, 
                      maturity: float, 
                      n_paths: int, 
                      n_steps: int,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo paths using the SABR model.
        
        Args:
            spot: Initial spot price
            maturity: Time to maturity in years
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (asset_paths, volatility_paths) as numpy arrays
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        asset_paths = np.zeros((n_paths, n_steps + 1))
        volatility_paths = np.zeros((n_paths, n_steps + 1))
        
        # Set initial values
        asset_paths[:, 0] = spot
        volatility_paths[:, 0] = self.alpha
        
        # Generate correlated random numbers
        z1 = np.random.standard_normal((n_paths, n_steps))
        z2 = np.random.standard_normal((n_paths, n_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
        
        # Generate paths
        for i in range(n_steps):
            # Current values
            F_t = asset_paths[:, i]
            alpha_t = volatility_paths[:, i]
            
            # Ensure positive values
            F_t = np.maximum(F_t, 1e-8)
            alpha_t = np.maximum(alpha_t, 1e-8)
            
            # Asset price evolution (CEV process)
            if self.beta == 1:
                # Log-normal case
                drift_F = 0.0
                diffusion_F = alpha_t * sqrt_dt * z1[:, i]
                asset_paths[:, i + 1] = F_t * np.exp(drift_F + diffusion_F)
            else:
                # CEV case
                diffusion_F = alpha_t * (F_t ** self.beta) * sqrt_dt * z1[:, i]
                asset_paths[:, i + 1] = F_t + diffusion_F
            
            # Volatility evolution
            diffusion_alpha = self.nu * alpha_t * sqrt_dt * z2[:, i]
            volatility_paths[:, i + 1] = np.maximum(alpha_t + diffusion_alpha, 1e-8)
        
        return asset_paths, volatility_paths
    
    def get_parameters(self) -> dict:
        """Get the current model parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'nu': self.nu,
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
        Validate SABR model parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # Check parameter bounds
        if (self.alpha <= 0 or self.beta < 0 or self.beta > 1 or 
            self.nu <= 0 or abs(self.rho) > 1):
            return False
        
        return True
    
    def get_analytical_price(self, 
                           spot: float,
                           strike: float,
                           maturity: float,
                           option_type: str = "call") -> float:
        """
        Calculate option price using SABR approximation.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            maturity: Time to maturity in years
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        try:
            # Calculate SABR implied volatility
            sabr_iv = self._calculate_sabr_implied_volatility(spot, strike, maturity)
            
            # Use Black-Scholes with SABR implied volatility
            return self._black_scholes_price(spot, strike, maturity, sabr_iv, option_type)
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate SABR price: {e}")
    
    def _calculate_sabr_implied_volatility(self, 
                                         spot: float, 
                                         strike: float, 
                                         maturity: float) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            maturity: Time to maturity in years
            
        Returns:
            Implied volatility
        """
        # Hagan's SABR implied volatility formula
        f = spot
        k = strike
        
        if abs(f - k) < 1e-8:
            # At-the-money case
            x = 1.0
            log_x = 0.0
        else:
            x = np.sqrt(f * k)
            log_x = np.log(f / k)
        
        # SABR parameters
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu
        
        # Calculate implied volatility
        if abs(beta - 1.0) < 1e-8:
            # Log-normal case
            c1 = 1.0
            c2 = 1.0
        else:
            c1 = (1.0 - beta) * log_x / (1.0 - beta)
            c2 = (1.0 - beta) * log_x / (1.0 - beta)
        
        # Hagan's formula
        term1 = alpha / (x ** (1.0 - beta))
        term2 = 1.0 + ((1.0 - beta) ** 2) * (log_x ** 2) / 24.0
        term3 = ((1.0 - beta) ** 4) * (log_x ** 4) / 1920.0
        
        # Correlation term
        corr_term = 0.25 * rho * nu * beta * alpha / (x ** (1.0 - beta))
        
        # Volatility of volatility term
        vol_term = (2.0 - 3.0 * rho ** 2) * (nu ** 2) / 24.0
        
        # Final implied volatility
        iv = term1 * (term2 + term3) * (1.0 + corr_term + vol_term)
        
        return iv
    
    def _black_scholes_price(self, 
                            spot: float,
                            strike: float,
                            maturity: float,
                            volatility: float,
                            option_type: str = "call") -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            maturity: Time to maturity in years
            volatility: Implied volatility
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        # Black-Scholes parameters
        S = spot
        K = strike
        T = maturity
        r = self.risk_free_rate
        sigma = volatility
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == "call":
            price = S * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
        else:
            price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
        
        return price
    
    def _normal_cdf(self, x: float) -> float:
        """Calculate standard normal cumulative distribution function."""
        return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))
    
    def get_implied_volatility_surface(self, 
                                     spot: float,
                                     strikes: np.ndarray,
                                     maturities: np.ndarray,
                                     option_type: str = "call") -> np.ndarray:
        """
        Calculate implied volatility surface for given strikes and maturities.
        
        Args:
            spot: Current spot price
            strikes: Array of strike prices
            maturities: Array of maturities in years
            option_type: "call" or "put"
            
        Returns:
            2D array of implied volatilities
        """
        iv_surface = np.zeros((len(strikes), len(maturities)))
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                try:
                    iv = self._calculate_sabr_implied_volatility(spot, strike, maturity)
                    iv_surface[i, j] = iv
                except Exception:
                    iv_surface[i, j] = np.nan
        
        return iv_surface
