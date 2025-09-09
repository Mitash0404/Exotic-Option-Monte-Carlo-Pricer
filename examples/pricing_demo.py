"""
Example: Comprehensive option pricing demonstration
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.heston_model import HestonModel
from models.sabr_model import SABRModel
from pricing.monte_carlo_engine import MonteCarloEngine
from pricing.payoff_functions import EuropeanPayoff, AsianPayoff, BarrierPayoff
from calibration.genetic_algorithm import GeneticAlgorithmCalibrator
from risk.delta_hedging import DeltaHedger
from risk.stress_testing import StressTester


def run_pricing_demo():
    """Run comprehensive pricing demonstration."""
    print("=== Exotic Option Monte Carlo Pricer Demo ===\n")
    
    # Initialize models
    print("1. Initializing Stochastic Volatility Models...")
    
    # Heston model
    heston_model = HestonModel(
        v0=0.04,      # Initial variance
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-term variance
        rho=-0.7,     # Correlation
        sigma=0.5,    # Volatility of volatility
        risk_free_rate=0.05
    )
    print(f"   Heston Model: v0={heston_model.v0}, kappa={heston_model.kappa}")
    
    # SABR model
    sabr_model = SABRModel(
        alpha=0.2,    # Initial volatility
        beta=0.5,     # CEV parameter
        rho=-0.1,     # Correlation
        nu=0.5,       # Volatility of volatility
        risk_free_rate=0.05
    )
    print(f"   SABR Model: alpha={sabr_model.alpha}, beta={sabr_model.beta}")
    
    # Initialize Monte Carlo engine
    print("\n2. Setting up Monte Carlo Engine...")
    mc_engine = MonteCarloEngine(
        model=heston_model,
        n_paths=100000,
        n_steps=252,
        use_antithetic=True,
        use_control_variates=True,
        seed=42
    )
    print(f"   Monte Carlo Engine: {mc_engine.n_paths:,} paths, {mc_engine.n_steps} steps")
    
    # Market parameters
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    
    print(f"\n3. Pricing Options (Spot: {spot}, Strike: {strike}, Maturity: {maturity} years)")
    print("   " + "="*60)
    
    # Price different option types
    option_types = [
        ("European Call", lambda: mc_engine.price_european_call(spot, strike, maturity)),
        ("European Put", lambda: mc_engine.price_european_put(spot, strike, maturity)),
        ("Asian Call (Arithmetic)", lambda: mc_engine.price_asian_option(spot, strike, maturity, "call", "arithmetic")),
        ("Asian Put (Arithmetic)", lambda: mc_engine.price_asian_option(spot, strike, maturity, "put", "arithmetic")),
        ("Barrier Call (Down-and-Out)", lambda: mc_engine.price_barrier_option(spot, strike, maturity, 90.0, "down-and-out", "call")),
        ("Barrier Put (Up-and-Out)", lambda: mc_engine.price_barrier_option(spot, strike, maturity, 110.0, "up-and-out", "put"))
    ]
    
    results = {}
    for option_name, pricing_func in option_types:
        print(f"\n   {option_name}:")
        start_time = time.time()
        price = pricing_func()
        pricing_time = time.time() - start_time
        
        print(f"     Price: ${price:.4f}")
        print(f"     Pricing Time: {pricing_time:.3f} seconds")
        results[option_name] = price
    
    # Accuracy analysis
    print("\n4. Pricing Accuracy Analysis...")
    print("   " + "="*60)
    
    # Compare with analytical solutions where available
    try:
        heston_analytical_call = heston_model.get_analytical_price(spot, strike, maturity, "call")
        heston_analytical_put = heston_model.get_analytical_price(spot, strike, maturity, "put")
        
        print(f"\n   European Call:")
        print(f"     Monte Carlo: ${results['European Call']:.4f}")
        print(f"     Analytical:  ${heston_analytical_call:.4f}")
        error = abs(results['European Call'] - heston_analytical_call) / heston_analytical_call
        print(f"     Error:       {error:.2%}")
        
        print(f"\n   European Put:")
        print(f"     Monte Carlo: ${results['European Put']:.4f}")
        print(f"     Analytical:  ${heston_analytical_put:.4f}")
        error = abs(results['European Put'] - heston_analytical_put) / heston_analytical_put
        print(f"     Error:       {error:.2%}")
        
    except Exception as e:
        print(f"   Analytical pricing not available: {e}")
    
    # Calibration demonstration
    print("\n5. Model Calibration...")
    print("   " + "="*60)
    
    # Generate mock market data
    market_strikes = np.array([90, 95, 100, 105, 110])
    market_prices = np.array([15.2, 12.1, 9.8, 7.9, 6.3])
    
    print(f"   Market Data: {len(market_strikes)} strikes")
    print(f"   Target Prices: {market_prices}")
    
    # Genetic algorithm calibration
    calibrator = GeneticAlgorithmCalibrator()
    print("\n   Running Genetic Algorithm Calibration...")
    start_time = time.time()
    
    calibrated_params = calibrator.calibrate(market_strikes, market_prices, heston_model)
    calibration_time = time.time() - start_time
    
    print(f"   Calibration Time: {calibration_time:.3f} seconds")
    print(f"   Calibrated Parameters:")
    for param, value in calibrated_params.items():
        print(f"     {param}: {value:.4f}")
    
    # Risk management demonstration
    print("\n6. Risk Management Analysis...")
    print("   " + "="*60)
    
    # Delta hedging
    hedger = DeltaHedger()
    print("   Delta Hedging Analysis:")
    
    # Calculate delta for European call
    delta = hedger.calculate_delta(heston_model, spot, strike, maturity, "call")
    print(f"     Delta (European Call): {delta:.4f}")
    
    # Stress testing
    stress_tester = StressTester()
    print("\n   Stress Testing (1000 scenarios):")
    
    stress_results = stress_tester.run_stress_test(
        heston_model, spot, strike, maturity, n_scenarios=1000
    )
    
    print(f"     VaR (95%): ${stress_results['var_95']:.2f}")
    print(f"     CVaR (95%): ${stress_results['cvar_95']:.2f}")
    print(f"     Max Loss: ${stress_results['max_loss']:.2f}")
    
    # Performance summary
    print("\n7. Performance Summary...")
    print("   " + "="*60)
    
    total_pricing_time = sum([
        time.time() - start_time for start_time, _ in [
            (time.time(), mc_engine.price_european_call(spot, strike, maturity))
        ]
    ])
    
    print(f"   Total Options Priced: {len(option_types)}")
    print(f"   Average Pricing Time: {total_pricing_time/len(option_types):.3f} seconds")
    print(f"   Monte Carlo Paths: {mc_engine.n_paths:,}")
    print(f"   Variance Reduction: Antithetic + Control Variates")
    
    print("\n=== Demo Completed Successfully! ===")
    return results


if __name__ == "__main__":
    results = run_pricing_demo()
