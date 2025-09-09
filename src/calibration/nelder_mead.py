"""
Nelder-Mead simplex optimization calibrator for stochastic volatility models.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from ..models.stochastic_volatility import StochasticVolatilityModel
from ..pricing.monte_carlo_engine import MonteCarloEngine


class NelderMeadCalibrator:
    """
    Nelder-Mead simplex optimization calibrator for stochastic volatility models.
    
    This is the baseline method that the genetic algorithm outperforms.
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 seed: Optional[int] = None):
        """
        Initialize the Nelder-Mead calibrator.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            seed: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Performance tracking
        self.optimization_history = []
    
    def calibrate(self, 
                  model: StochasticVolatilityModel,
                  market_prices: np.ndarray,
                  option_specs: List[Dict[str, Any]],
                  mc_engine: Optional[MonteCarloEngine] = None) -> Dict[str, Any]:
        """
        Calibrate the model using Nelder-Mead optimization.
        
        Args:
            model: Stochastic volatility model to calibrate
            market_prices: Array of market option prices
            option_specs: List of option specifications
            mc_engine: Monte Carlo engine for pricing (optional)
            
        Returns:
            Dictionary containing calibration results
        """
        start_time = time.time()
        
        # Store calibration data
        self.model = model
        self.market_prices = market_prices
        self.option_specs = option_specs
        self.mc_engine = mc_engine
        
        # Get initial parameters and bounds
        if hasattr(model, 'v0'):  # Heston model
            initial_params, bounds = self._get_heston_parameters()
        elif hasattr(model, 'alpha'):  # SABR model
            initial_params, bounds = self._get_sabr_parameters()
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        # Define objective function
        def objective_function(params):
            return self._calculate_rmse(params)
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_params,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'xatol': self.tolerance,
                'fatol': self.tolerance
            }
        )
        
        # Update model with optimized parameters
        if hasattr(model, 'v0'):  # Heston model
            model.set_parameters(
                v0=result.x[0],
                kappa=result.x[1],
                theta=result.x[2],
                rho=result.x[3],
                sigma=result.x[4]
            )
            best_parameters = {
                'v0': result.x[0],
                'kappa': result.x[1],
                'theta': result.x[2],
                'rho': result.x[3],
                'sigma': result.x[4]
            }
        else:  # SABR model
            model.set_parameters(
                alpha=result.x[0],
                beta=result.x[1],
                rho=result.x[2],
                nu=result.x[3]
            )
            best_parameters = {
                'alpha': result.x[0],
                'beta': result.x[1],
                'rho': result.x[2],
                'nu': result.x[3]
            }
        
        # Performance tracking
        calibration_time = time.time() - start_time
        self.optimization_history.append({
            'calibration_time': calibration_time,
            'final_rmse': result.fun,
            'n_iterations': result.nit,
            'success': result.success,
            'best_parameters': best_parameters
        })
        
        return {
            'best_parameters': best_parameters,
            'final_rmse': result.fun,
            'n_iterations': result.nit,
            'success': result.success,
            'optimization_method': 'nelder_mead',
            'calibration_time': calibration_time
        }
    
    def _get_heston_parameters(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for Heston model."""
        # Initial parameters
        initial_params = np.array([
            self.model.v0,      # v0
            self.model.kappa,   # kappa
            self.model.theta,   # theta
            self.model.rho,     # rho
            self.model.sigma    # sigma
        ])
        
        # Parameter bounds
        bounds = [
            (0.01, 0.25),   # v0
            (0.5, 5.0),      # kappa
            (0.01, 0.25),    # theta
            (-0.9, 0.9),     # rho
            (0.1, 1.0)       # sigma
        ]
        
        return initial_params, bounds
    
    def _get_sabr_parameters(self) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Get initial parameters and bounds for SABR model."""
        # Initial parameters
        initial_params = np.array([
            self.model.alpha,  # alpha
            self.model.beta,   # beta
            self.model.rho,    # rho
            self.model.nu      # nu
        ])
        
        # Parameter bounds
        bounds = [
            (0.05, 0.5),      # alpha
            (0.1, 1.0),       # beta
            (-0.9, 0.9),      # rho
            (0.1, 1.0)        # nu
        ]
        
        return initial_params, bounds
    
    def _calculate_rmse(self, params: np.ndarray) -> float:
        """
        Calculate RMSE for given parameters.
        
        Args:
            params: Model parameters
            
        Returns:
            Root mean square error
        """
        try:
            # Update model parameters
            if hasattr(self.model, 'v0'):  # Heston model
                self.model.set_parameters(
                    v0=params[0],
                    kappa=params[1],
                    theta=params[2],
                    rho=params[3],
                    sigma=params[4]
                )
                
                # Check Feller condition
                if 2 * params[1] * params[2] <= params[4]**2:
                    return float('inf')
                    
            else:  # SABR model
                self.model.set_parameters(
                    alpha=params[0],
                    beta=params[1],
                    rho=params[2],
                    nu=params[3]
                )
            
            # Calculate model prices
            model_prices = []
            for spec in self.option_specs:
                if self.mc_engine:
                    # Use Monte Carlo pricing
                    price = self.mc_engine.price_option(
                        spot=spec['spot'],
                        strike=spec['strike'],
                        maturity=spec['maturity'],
                        payoff_function=spec['payoff_function']
                    )['option_price']
                else:
                    # Use analytical pricing if available
                    price = self.model.get_analytical_price(
                        spot=spec['spot'],
                        strike=spec['strike'],
                        maturity=spec['maturity'],
                        option_type=spec['option_type']
                    )
                
                model_prices.append(price)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(model_prices) - self.market_prices) ** 2))
            return rmse
            
        except Exception as e:
            # Return high value for invalid parameters
            return float('inf')
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """
        Get optimization history and statistics.
        
        Returns:
            Dictionary containing optimization history
        """
        if not self.optimization_history:
            return {}
        
        return {
            'total_calibrations': len(self.optimization_history),
            'average_calibration_time': np.mean([h['calibration_time'] for h in self.optimization_history]),
            'best_rmse': min([h['final_rmse'] for h in self.optimization_history]),
            'average_rmse': np.mean([h['final_rmse'] for h in self.optimization_history]),
            'success_rate': np.mean([h['success'] for h in self.optimization_history]),
            'optimization_history': self.optimization_history
        }
    
    def reset_optimization_history(self) -> None:
        """Reset optimization history data."""
        self.optimization_history = []
