"""
Exotic Option Monte Carlo Pricer

A sophisticated Monte Carlo pricing engine for exotic options with genetic algorithm calibration
and comprehensive risk management.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"
__email__ = "team@quantfinance.com"

from .models import HestonModel, SABRModel
from .pricing import MonteCarloEngine, PayoffFunction, EuropeanPayoff

__all__ = [
    "HestonModel",
    "SABRModel", 
    "MonteCarloEngine",
    "PayoffFunction",
    "EuropeanPayoff"
]
