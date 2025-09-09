#!/usr/bin/env python3
"""
Demo script for the Exotic Option Monte Carlo Pricer.
"""

from src.models.heston_model import HestonModel
from src.pricing.monte_carlo_engine import MonteCarloEngine
import time

def demo_monte_carlo_pricing():
    """Demonstrate Monte Carlo pricing with the Heston model."""
    print("=" * 80)
    print("EXOTIC OPTION MONTE CARLO PRICER - DEMO")
    print("=" * 80)
    
    # Initialize Heston model
    print("Initializing Heston stochastic volatility model...")
    heston = HestonModel(
        v0=0.04,      # Initial variance
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-term variance
        rho=-0.7,     # Correlation
        sigma=0.3,    # Volatility of volatility (reduced to satisfy Feller condition)
        risk_free_rate=0.05
    )
    print(f"✓ Model created: {heston}")
    
    # Create Monte Carlo engine
    print("\nSetting up Monte Carlo pricing engine...")
    mc_engine = MonteCarloEngine(
        model=heston,
        n_paths=50000,  # Number of Monte Carlo paths
        n_steps=252,    # Daily steps (1 year)
        seed=42         # For reproducibility
    )
    print(f"✓ Engine created: {mc_engine}")
    
    # Test parameters
    spot_price = 100.0
    strike_price = 100.0
    maturity = 1.0  # 1 year
    
    print(f"\nPricing European call option:")
    print(f"  Spot: ${spot_price}")
    print(f"  Strike: ${strike_price}")
    print(f"  Maturity: {maturity} years")
    print(f"  Risk-free rate: {heston.risk_free_rate:.1%}")
    
    # Price the option
    start_time = time.time()
    option_price = mc_engine.price_european_call(
        spot=spot_price,
        strike=strike_price,
        maturity=maturity
    )
    pricing_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Option Price: ${option_price:.4f}")
    print(f"  Pricing Time: {pricing_time:.3f} seconds")
    
    # Performance statistics
    stats = mc_engine.get_performance_stats()
    if stats:
        print(f"\nPerformance Statistics:")
        print(f"  Total Pricings: {stats['total_pricings']}")
        print(f"  Average Time: {stats['average_pricing_time']:.3f}s")
    
    print("\n🎉 Demo completed successfully!")
    return option_price

if __name__ == "__main__":
    try:
        demo_monte_carlo_pricing()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
