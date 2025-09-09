"""
Genetic algorithm calibrator for stochastic volatility models.
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
from ..models.stochastic_volatility import StochasticVolatilityModel


class GeneticAlgorithmCalibrator:
    """
    Genetic algorithm calibrator for model parameter optimization.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 30,
                 mutation_rate: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the genetic algorithm calibrator.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def calibrate(self, 
                  model: StochasticVolatilityModel,
                  market_prices: List[float],
                  option_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate model parameters using genetic algorithm.
        """
        start_time = time.time()
        
        # Get initial parameters and bounds
        initial_params = model.get_parameters()
        param_bounds = self._get_parameter_bounds(model)
        
        # Initialize population
        population = self._initialize_population(param_bounds)
        
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, model, market_prices, option_specs)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            fitness_history.append(best_fitness)
            
            # Selection and reproduction
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, param_bounds)
                new_population.append(child)
            
            population = new_population
        
        calibration_time = time.time() - start_time
        
        # Set best parameters to model
        if best_individual is not None:
            self._set_model_parameters(model, best_individual, param_bounds)
        
        return {
            'best_parameters': self._individual_to_params(best_individual, param_bounds),
            'final_rmse': best_fitness,
            'calibration_time': calibration_time,
            'n_generations': self.generations,
            'fitness_history': fitness_history
        }
    
    def _get_parameter_bounds(self, model: StochasticVolatilityModel) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for the model."""
        if hasattr(model, 'v0'):  # Heston model
            return {
                'v0': (0.01, 0.20),
                'kappa': (0.5, 5.0),
                'theta': (0.01, 0.20),
                'rho': (-0.99, 0.99),
                'sigma': (0.1, 1.0)
            }
        else:  # SABR model
            return {
                'alpha': (0.05, 0.50),
                'beta': (0.1, 0.9),
                'rho': (-0.99, 0.99),
                'nu': (0.1, 1.0)
            }
    
    def _initialize_population(self, param_bounds: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = []
            for param_name, (low, high) in param_bounds.items():
                if param_name == 'rho':  # Correlation parameter
                    individual.append(np.random.uniform(low, high))
                else:
                    individual.append(np.random.uniform(low, high))
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, 
                         individual: List[float], 
                         model: StochasticVolatilityModel,
                         market_prices: List[float],
                         option_specs: List[Dict[str, Any]]) -> float:
        """Evaluate fitness (RMSE) of an individual."""
        try:
            # Set model parameters
            param_bounds = self._get_parameter_bounds(model)
            params = self._individual_to_params(individual, param_bounds)
            model.set_parameters(**params)
            
            # Calculate model prices
            model_prices = []
            for spec in option_specs:
                if spec['type'] == 'european':
                    if spec['option_type'] == 'call':
                        price = model.mc_engine.price_european_call(
                            spec['spot'], spec['strike'], spec['maturity']
                        )
                    else:
                        price = model.mc_engine.price_european_put(
                            spec['spot'], spec['strike'], spec['maturity']
                        )
                else:
                    price = 0.0  # Placeholder for other option types
                
                model_prices.append(price)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(model_prices) - np.array(market_prices))**2))
            return rmse
            
        except Exception:
            return float('inf')  # Penalty for invalid parameters
    
    def _tournament_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[float]:
        """Tournament selection."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def _mutate(self, individual: List[float], param_bounds: Dict[str, Tuple[float, float]]) -> List[float]:
        """Gaussian mutation."""
        mutated = individual.copy()
        for i, (param_name, (low, high)) in enumerate(param_bounds.items()):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                sigma = (high - low) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated
    
    def _individual_to_params(self, individual: List[float], param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Convert individual to parameter dictionary."""
        params = {}
        for i, (param_name, _) in enumerate(param_bounds.items()):
            params[param_name] = individual[i]
        return params
    
    def _set_model_parameters(self, model: StochasticVolatilityModel, individual: List[float], param_bounds: Dict[str, Tuple[float, float]]):
        """Set model parameters from individual."""
        params = self._individual_to_params(individual, param_bounds)
        model.set_parameters(**params)
