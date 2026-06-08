import logging
import sys
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
)

logger = logging.getLogger("main")

from mc_pricing.gbm_kernel import monte_carlo_european_gpu, monte_carlo_european_cpu
from mc_pricing.black_scholes import black_scholes_price
from mc_pricing.implied_vol import newton_raphson_iv
from mc_pricing.market_data import fetch_option_chain, generate_synthetic_market_data
from mc_pricing.vol_surface import VolatilitySurface
from mc_pricing.visualization import (
    plot_volatility_surface_3d,
    plot_mc_convergence,
    plot_price_distribution,
    plot_volatility_smile_slices,
)


def run_monte_carlo_pricing():
    logger.info("=" * 70)
    logger.info("  PHASE 1: GPU-Accelerated Monte Carlo European Option Pricing")
    logger.info("=" * 70)

    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    N_PATHS = 1_000_000
    N_STEPS = 252

    bs_call = black_scholes_price(S0, K, T, r, sigma, 'call')
    bs_put = black_scholes_price(S0, K, T, r, sigma, 'put')
    logger.info(f"Black-Scholes Call: {bs_call:.6f}  |  Put: {bs_put:.6f}")

    mc_call = monte_carlo_european_gpu(
        S0, K, T, r, sigma,
        n_paths=N_PATHS, n_steps=N_STEPS,
        option_type='call', rng_seed=42,
    )

    mc_put = monte_carlo_european_gpu(
        S0, K, T, r, sigma,
        n_paths=N_PATHS, n_steps=N_STEPS,
        option_type='put', rng_seed=43,
    )

    call_err = abs(mc_call['price'] - bs_call)
    put_err = abs(mc_put['price'] - bs_put)
    call_rel = call_err / bs_call * 100
    put_rel = put_err / bs_put * 100

    logger.info("-" * 70)
    logger.info(f"  CALL  │ MC: {mc_call['price']:.6f}  │ BS: {bs_call:.6f}  │ "
                f"Abs Err: {call_err:.6f}  │ Rel Err: {call_rel:.4f}%  │ "
                f"Time: {mc_call['elapsed_ms']:.2f}ms [{mc_call['device']}]")
    logger.info(f"  PUT   │ MC: {mc_put['price']:.6f}  │ BS: {bs_put:.6f}  │ "
                f"Abs Err: {put_err:.6f}  │ Rel Err: {put_rel:.4f}%  │ "
                f"Time: {mc_put['elapsed_ms']:.2f}ms [{mc_put['device']}]")
    logger.info("-" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig_conv = plot_mc_convergence(mc_call, bs_call, title="Call Option — MC vs Black-Scholes")
    fig_conv.write_html(os.path.join(output_dir, 'mc_convergence_call.html'))
    logger.info("Saved: mc_convergence_call.html")

    fig_dist = plot_price_distribution(mc_call, S0, K, T, r, sigma, 'call')
    fig_dist.write_html(os.path.join(output_dir, 'price_distribution_call.html'))
    logger.info("Saved: price_distribution_call.html")

    return mc_call, mc_put, bs_call, bs_put


def run_implied_volatility_solver():
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 2: Newton-Raphson Implied Volatility Solver")
    logger.info("=" * 70)

    S0 = 100.0
    K = 100.0
    T = 0.5
    r = 0.05
    true_sigma = 0.25

    bs_price = black_scholes_price(S0, K, T, r, true_sigma, 'call')
    logger.info(f"True σ = {true_sigma:.4f}  →  BS Price = {bs_price:.6f}")

    recovered_iv = newton_raphson_iv(bs_price, S0, K, T, r, 'call')
    logger.info(f"Newton-Raphson recovered σ = {recovered_iv:.8f}")
    logger.info(f"Absolute error: {abs(recovered_iv - true_sigma):.10f}")

    logger.info("\n  Implied Vol across different moneyness levels:")
    logger.info(f"  {'Moneyness':>10s}  {'K':>8s}  {'BS Price':>10s}  │  {'Recovered IV':>14s}  {'True IV':>10s}  {'Error':>12s}")
    logger.info("  " + "-" * 70)

    for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
        K_test = S0 / m
        skew = 0.10 * (m - 1.0) ** 2
        true_iv = true_sigma + skew
        p = black_scholes_price(S0, K_test, T, r, true_iv, 'call')
        iv = newton_raphson_iv(p, S0, K_test, T, r, 'call')
        err = abs(iv - true_iv)
        logger.info(f"  {m:>10.2f}  {K_test:>8.1f}  {p:>10.4f}  │  {iv:>14.8f}  {true_iv:>10.6f}  {err:>12.10f}")

    return recovered_iv


def run_volatility_surface():
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 3: 3D Volatility Surface Construction & Visualization")
    logger.info("=" * 70)

    logger.info("Fetching market data...")
    try:
        df = fetch_option_chain('SPY', max_expiries=6)
        if df['ticker'].iloc[0] == 'SYNTHETIC':
            logger.info("Using synthetic market data (yfinance unavailable or no data)")
        else:
            logger.info(f"Using real market data for {df['ticker'].iloc[0]}")
    except Exception as e:
        logger.warning(f"Data fetch failed: {e}, using synthetic data")
        df = generate_synthetic_market_data()

    r = 0.05
    vs = VolatilitySurface(df, r=r, option_type='call')

    logger.info("Computing implied volatilities via Newton-Raphson...")
    vs.compute_implied_volatilities()

    logger.info("Building cubic spline volatility surface...")
    vs.build_surface(n_strike_points=100, n_expiry_points=50)

    surface_data = vs.get_surface_arrays()
    logger.info(f"Surface grid: {len(surface_data['strikes'])} strikes × {len(surface_data['expiries'])} expiries")
    logger.info(f"IV range: {surface_data['iv_surface'].min()*100:.2f}% — {surface_data['iv_surface'].max()*100:.2f}%")

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    fig_surface = plot_volatility_surface_3d(
        surface_data,
        title="Implied Volatility Surface — Volatility Smile",
        show_scatter=True,
        scatter_data=vs.raw_data,
    )
    fig_surface.write_html(os.path.join(output_dir, 'volatility_surface_3d.html'))
    logger.info("Saved: volatility_surface_3d.html")

    fig_slices = plot_volatility_smile_slices(
        surface_data,
        slices_at_days=[30, 90, 180, 270, 365],
        title="Volatility Smile — Expiry Slices",
    )
    fig_slices.write_html(os.path.join(output_dir, 'volatility_smile_slices.html'))
    logger.info("Saved: volatility_smile_slices.html")

    return vs


def main():
    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║   Options Pricing & Volatility Surface Analysis Platform            ║")
    logger.info("║   GPU-Accelerated Monte Carlo │ Newton-Raphson IV │ 3D Visualization ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info("")

    mc_call, mc_put, bs_call, bs_put = run_monte_carlo_pricing()
    recovered_iv = run_implied_volatility_solver()
    vs = run_volatility_surface()

    logger.info("")
    logger.info("=" * 70)
    logger.info("  ALL PHASES COMPLETE — Output files saved to ./output/")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Generated Files:")
    logger.info("    📊 mc_convergence_call.html     — MC vs BS convergence dashboard")
    logger.info("    📊 price_distribution_call.html  — Terminal price & payoff distributions")
    logger.info("    📊 volatility_surface_3d.html    — 3D implied volatility surface")
    logger.info("    📊 volatility_smile_slices.html  — Volatility smile cross-sections")
    logger.info("")
    logger.info("  Open the HTML files in a browser to explore interactive visualizations.")


if __name__ == '__main__':
    main()
