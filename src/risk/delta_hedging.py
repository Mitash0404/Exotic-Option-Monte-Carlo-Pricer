"""
Delta hedging implementation for option risk management.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..models.stochastic_volatility import StochasticVolatilityModel
from ..pricing.monte_carlo_engine import MonteCarloEngine


@dataclass
class HedgePosition:
    """Represents a hedge position."""
    option_position: float  # Number of options
    underlying_position: float  # Number of underlying shares
    cash_position: float  # Cash position
    timestamp: float  # Time from start
    spot_price: float  # Current spot price
    delta: float  # Option delta
    gamma: float  # Option gamma
    theta: float  # Option theta
    vega: float  # Option vega


class DeltaHedger:
    """
    Delta hedging implementation for option risk management.
    
    Implements dynamic delta hedging with rebalancing at specified intervals.
    """
    
    def __init__(self, 
                 model: StochasticVolatilityModel,
                 mc_engine: MonteCarloEngine,
                 rebalance_frequency: float = 1/252,  # Daily rebalancing
                 transaction_cost: float = 0.001,  # 0.1% transaction cost
                 initial_capital: float = 1000000.0):
        """
        Initialize the delta hedger.
        
        Args:
            model: Stochastic volatility model
            mc_engine: Monte Carlo pricing engine
            rebalance_frequency: Time between rebalancing (in years)
            transaction_cost: Transaction cost as fraction of trade value
            initial_capital: Initial capital for hedging
        """
        self.model = model
        self.mc_engine = mc_engine
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        # Hedging history
        self.hedge_positions: List[HedgePosition] = []
        self.portfolio_values: List[float] = []
        self.hedge_costs: List[float] = []
        
    def hedge_option(self, 
                    option_spec: Dict[str, Any],
                    n_options: int = 1,
                    hedge_duration: float = 1.0,
                    n_scenarios: int = 1000) -> Dict[str, Any]:
        """
        Delta hedge an option position.
        
        Args:
            option_spec: Option specification
            n_options: Number of options to hedge
            hedge_duration: Duration of hedging period (years)
            n_scenarios: Number of market scenarios to simulate
            
        Returns:
            Dictionary containing hedging results
        """
        # Extract option parameters
        spot = option_spec['spot']
        strike = option_spec['strike']
        maturity = option_spec['maturity']
        option_type = option_spec['option_type']
        payoff_function = option_spec['payoff_function']
        
        # Initialize hedging
        current_time = 0.0
        current_spot = spot
        current_capital = self.initial_capital
        
        # Calculate initial option price and Greeks
        initial_pricing = self.mc_engine.price_option(
            spot=current_spot,
            strike=strike,
            maturity=maturity - current_time,
            payoff_function=payoff_function
        )
        
        initial_option_price = initial_pricing['option_price']
        initial_delta = self._calculate_delta(
            spot=current_spot,
            strike=strike,
            maturity=maturity - current_time,
            option_type=option_type
        )
        
        # Initial positions
        option_cost = n_options * initial_option_price
        required_hedge = -n_options * initial_delta
        hedge_cost = abs(required_hedge) * current_spot * (1 + self.transaction_cost)
        
        # Check if we have enough capital
        if option_cost + hedge_cost > current_capital:
            raise ValueError("Insufficient capital for hedging")
        
        # Update capital
        current_capital -= option_cost + hedge_cost
        
        # Record initial position
        initial_position = HedgePosition(
            option_position=n_options,
            underlying_position=required_hedge,
            cash_position=current_capital,
            timestamp=current_time,
            spot_price=current_spot,
            delta=initial_delta,
            gamma=self._calculate_gamma(current_spot, strike, maturity - current_time, option_type),
            theta=self._calculate_theta(current_spot, strike, maturity - current_time, option_type),
            vega=self._calculate_vega(current_spot, strike, maturity - current_time, option_type)
        )
        
        self.hedge_positions.append(initial_position)
        self.portfolio_values.append(self._calculate_portfolio_value(initial_position, current_spot))
        self.hedge_costs.append(hedge_cost)
        
        # Simulate hedging over time
        while current_time < hedge_duration:
            # Advance time
            current_time += self.rebalance_frequency
            if current_time > maturity:
                current_time = maturity
            
            # Simulate spot price movement
            current_spot = self._simulate_spot_movement(
                current_spot, 
                self.rebalance_frequency,
                n_scenarios
            )
            
            # Calculate new Greeks
            time_to_maturity = max(0, maturity - current_time)
            if time_to_maturity > 0:
                new_delta = self._calculate_delta(current_spot, strike, time_to_maturity, option_type)
                new_gamma = self._calculate_gamma(current_spot, strike, time_to_maturity, option_type)
                new_theta = self._calculate_theta(current_spot, strike, time_to_maturity, option_type)
                new_vega = self._calculate_vega(current_spot, strike, time_to_maturity, option_type)
            else:
                # Option has expired
                new_delta = new_gamma = new_theta = new_vega = 0.0
            
            # Calculate required hedge adjustment
            required_hedge = -n_options * new_delta
            hedge_adjustment = required_hedge - initial_position.underlying_position
            
            # Calculate transaction costs
            if abs(hedge_adjustment) > 1e-6:  # Only rebalance if significant change
                transaction_cost = abs(hedge_adjustment) * current_spot * self.transaction_cost
                current_capital -= transaction_cost
                self.hedge_costs.append(transaction_cost)
            else:
                self.hedge_costs.append(0.0)
            
            # Update position
            new_position = HedgePosition(
                option_position=n_options,
                underlying_position=required_hedge,
                cash_position=current_capital,
                timestamp=current_time,
                spot_price=current_spot,
                delta=new_delta,
                gamma=new_gamma,
                theta=new_theta,
                vega=new_vega
            )
            
            self.hedge_positions.append(new_position)
            self.portfolio_values.append(self._calculate_portfolio_value(new_position, current_spot))
            
            # Update initial position for next iteration
            initial_position = new_position
        
        # Calculate final portfolio value
        final_portfolio_value = self._calculate_portfolio_value(
            self.hedge_positions[-1], 
            self.hedge_positions[-1].spot_price
        )
        
        # Calculate hedging performance metrics
        total_hedge_cost = sum(self.hedge_costs)
        portfolio_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        hedge_effectiveness = self._calculate_hedge_effectiveness()
        
        return {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_hedge_cost': total_hedge_cost,
            'portfolio_return': portfolio_return,
            'hedge_effectiveness': hedge_effectiveness,
            'n_rebalances': len(self.hedge_positions) - 1,
            'hedge_positions': self.hedge_positions,
            'portfolio_values': self.portfolio_values,
            'hedge_costs': self.hedge_costs
        }
    
    def _calculate_delta(self, 
                        spot: float, 
                        strike: float, 
                        maturity: float, 
                        option_type: str) -> float:
        """Calculate option delta using finite difference."""
        epsilon = spot * 0.001  # Small perturbation
        
        # Price at spot + epsilon
        price_up = self.mc_engine.price_option(
            spot=spot + epsilon,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Price at spot - epsilon
        price_down = self.mc_engine.price_option(
            spot=spot - epsilon,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Delta = (price_up - price_down) / (2 * epsilon)
        delta = (price_up - price_down) / (2 * epsilon)
        
        return delta
    
    def _calculate_gamma(self, 
                        spot: float, 
                        strike: float, 
                        maturity: float, 
                        option_type: str) -> float:
        """Calculate option gamma using finite difference."""
        epsilon = spot * 0.001  # Small perturbation
        
        # Price at current spot
        price_current = self.mc_engine.price_option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Price at spot + epsilon
        price_up = self.mc_engine.price_option(
            spot=spot + epsilon,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Price at spot - epsilon
        price_down = self.mc_engine.price_option(
            spot=spot - epsilon,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Gamma = (price_up + price_down - 2*price_current) / epsilon^2
        gamma = (price_up + price_down - 2 * price_current) / (epsilon ** 2)
        
        return gamma
    
    def _calculate_theta(self, 
                        spot: float, 
                        strike: float, 
                        maturity: float, 
                        option_type: str) -> float:
        """Calculate option theta using finite difference."""
        epsilon = 1/252  # Small time perturbation (1 day)
        
        # Price at current maturity
        price_current = self.mc_engine.price_option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Price at maturity + epsilon
        price_future = self.mc_engine.price_option(
            spot=spot,
            strike=strike,
            maturity=maturity + epsilon,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Theta = (price_future - price_current) / epsilon
        theta = (price_future - price_current) / epsilon
        
        return theta
    
    def _calculate_vega(self, 
                       spot: float, 
                       strike: float, 
                       maturity: float, 
                       option_type: str) -> float:
        """Calculate option vega using finite difference."""
        # Store original volatility parameters
        original_params = self.model.get_parameters()
        
        epsilon = 0.001  # Small volatility perturbation
        
        # Price at current volatility
        price_current = self.mc_engine.price_option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Temporarily increase volatility
        if hasattr(self.model, 'v0'):  # Heston model
            self.model.set_parameters(v0=original_params['v0'] + epsilon)
        elif hasattr(self.model, 'alpha'):  # SABR model
            self.model.set_parameters(alpha=original_params['alpha'] + epsilon)
        
        # Price at increased volatility
        price_up = self.mc_engine.price_option(
            spot=spot,
            strike=strike,
            maturity=maturity,
            payoff_function=self._create_payoff_function(strike, option_type)
        )['option_price']
        
        # Restore original parameters
        self.model.set_parameters(**original_params)
        
        # Vega = (price_up - price_current) / epsilon
        vega = (price_up - price_current) / epsilon
        
        return vega
    
    def _simulate_spot_movement(self, 
                               current_spot: float, 
                               dt: float, 
                               n_scenarios: int) -> float:
        """Simulate spot price movement using the stochastic volatility model."""
        # Generate a single step using the model
        asset_paths, _ = self.model.generate_paths(
            spot=current_spot,
            maturity=dt,
            n_paths=n_scenarios,
            n_steps=1,
            seed=None
        )
        
        # Return the average final price
        return np.mean(asset_paths[:, -1])
    
    def _calculate_portfolio_value(self, position: HedgePosition, spot_price: float) -> float:
        """Calculate current portfolio value."""
        option_value = position.option_position * self._calculate_option_value(
            position.spot_price, 
            position.timestamp
        )
        underlying_value = position.underlying_position * spot_price
        cash_value = position.cash_position
        
        return option_value + underlying_value + cash_value
    
    def _calculate_option_value(self, spot_price: float, timestamp: float) -> float:
        """Calculate current option value."""
        # This is a simplified calculation - in practice, you'd use the full option spec
        # For now, return a placeholder value
        return 0.0
    
    def _create_payoff_function(self, strike: float, option_type: str):
        """Create a payoff function for the given option type."""
        from ..pricing.payoff_functions import EuropeanPayoff
        return EuropeanPayoff(strike, option_type)
    
    def _calculate_hedge_effectiveness(self) -> float:
        """Calculate hedge effectiveness (correlation between option and hedge returns)."""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        # Calculate returns
        option_returns = []
        hedge_returns = []
        
        for i in range(1, len(self.hedge_positions)):
            # Option return (simplified)
            option_return = 0.0  # Placeholder
            
            # Hedge return
            hedge_return = (self.hedge_positions[i].underlying_position - 
                          self.hedge_positions[i-1].underlying_position) / \
                          abs(self.hedge_positions[i-1].underlying_position) if \
                          abs(self.hedge_positions[i-1].underlying_position) > 1e-6 else 0.0
            
            option_returns.append(option_return)
            hedge_returns.append(hedge_return)
        
        if len(option_returns) > 1:
            correlation = np.corrcoef(option_returns, hedge_returns)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def reset_hedging_history(self) -> None:
        """Reset hedging history data."""
        self.hedge_positions = []
        self.portfolio_values = []
        self.hedge_costs = []
