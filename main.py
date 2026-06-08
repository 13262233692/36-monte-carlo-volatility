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

from mc_pricing.gbm_kernel import (
    monte_carlo_european_gpu,
    monte_carlo_european_cpu,
    monte_carlo_basket_gpu,
    monte_carlo_basket_cpu,
    get_gpu_memory_info,
    compute_optimal_batch_size,
)
from mc_pricing.black_scholes import black_scholes_price
from mc_pricing.implied_vol import newton_raphson_iv
from mc_pricing.market_data import fetch_option_chain, generate_synthetic_market_data
from mc_pricing.vol_surface import VolatilitySurface
from mc_pricing.pde_solver import (
    crank_nicolson_american,
    crank_nicolson_european,
    dupire_local_volatility,
    compare_american_european,
)
from mc_pricing.visualization import (
    plot_volatility_surface_3d,
    plot_mc_convergence,
    plot_price_distribution,
    plot_volatility_smile_slices,
    plot_early_exercise_boundary,
    plot_pde_price_profile,
)


def run_gpu_memory_diagnostics():
    logger.info("=" * 70)
    logger.info("  PHASE 0: GPU Memory Diagnostics & Batch Sizing")
    logger.info("=" * 70)

    mem_info = get_gpu_memory_info()
    if mem_info['total_mb'] > 0:
        logger.info(f"  GPU VRAM: {mem_info['free_mb']:.0f}MB free / {mem_info['total_mb']:.0f}MB total "
                    f"({mem_info['used_mb']:.0f}MB used)")

        for n_assets in [1, 5, 10, 20, 50]:
            bs = compute_optimal_batch_size(n_assets=n_assets, n_steps=252, free_vram_mb=mem_info['free_mb'])
            vram_per_batch = bs * n_assets * 8 * 3 / (1024**2)
            logger.info(f"  n_assets={n_assets:>2d} → batch_size={bs:>7,} paths | "
                        f"working set ≈ {vram_per_batch:.1f}MB/batch-step")
    else:
        logger.info("  No GPU detected — CPU fallback mode active")
        logger.info("  Batch processing still applies on CPU (memory-adaptive)")

    return mem_info


def run_monte_carlo_pricing():
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 1: Batched Monte Carlo European Option Pricing")
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

    return mc_call, mc_put, bs_call, bs_put


def run_basket_option_pricing():
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 1B: Multi-Asset Basket Option — Batched GPU Pricing")
    logger.info("=" * 70)

    n_assets = 10
    np.random.seed(42)

    S0_vec = np.random.uniform(80, 150, n_assets).tolist()
    weights = np.ones(n_assets) / n_assets
    sigma_vec = np.random.uniform(0.15, 0.40, n_assets).tolist()

    base_corr = 0.4
    corr_matrix = base_corr * np.ones((n_assets, n_assets)) + (1 - base_corr) * np.eye(n_assets)

    K = np.dot(S0_vec, weights)
    T = 0.5
    r = 0.05
    N_PATHS = 1_000_000
    N_STEPS = 126

    logger.info(f"  Basket: {n_assets} assets | K={K:.2f} | T={T} | r={r}")
    logger.info(f"  S0 range: [{min(S0_vec):.1f}, {max(S0_vec):.1f}]")
    logger.info(f"  σ range: [{min(sigma_vec):.3f}, {max(sigma_vec):.3f}]")
    logger.info(f"  Correlation: base={base_corr}")
    logger.info(f"  Paths: {N_PATHS:,} | Steps: {N_STEPS}")

    mem_before = get_gpu_memory_info()

    basket_result = monte_carlo_basket_gpu(
        S0_vec, weights, K, T, r, sigma_vec, corr_matrix,
        n_paths=N_PATHS, n_steps=N_STEPS,
        option_type='call', rng_seed=42, antithetic=True,
    )

    mem_after = get_gpu_memory_info()

    logger.info("-" * 70)
    logger.info(f"  BASKET CALL │ Price: {basket_result['price']:.6f} │ "
                f"StdErr: {basket_result['std_error']:.6f} │ "
                f"Time: {basket_result['elapsed_ms']:.2f}ms [{basket_result['device']}]")
    if mem_after['total_mb'] > 0:
        vram_delta = mem_after['used_mb'] - mem_before['used_mb']
        logger.info(f"  VRAM delta: {vram_delta:+.0f}MB | "
                    f"Peak used: {mem_after['used_mb']:.0f}MB / {mem_after['total_mb']:.0f}MB")
    logger.info("-" * 70)

    memory_comparison(basket_result, N_PATHS, N_STEPS, n_assets)

    return basket_result


def memory_comparison(result, n_paths, n_steps, n_assets):
    logger.info("")
    logger.info("  ┌─────────────────────────────────────────────────────────────┐")
    logger.info("  │          MEMORY FOOTPRINT COMPARISON                        │")
    logger.info("  ├─────────────────────────────────────────────────────────────┤")

    legacy_rng_bytes = n_paths * n_steps * n_assets * 8
    legacy_full_bytes = n_paths * (n_steps + 1) * n_assets * 8
    legacy_total = (legacy_rng_bytes + legacy_full_bytes) / (1024**3)
    logger.info(f"  │ LEGACY (monolithic):                                       │")
    logger.info(f"  │   RNG pool:    {legacy_rng_bytes/(1024**3):.3f} GB  "
                f"({n_paths:,} × {n_steps} × {n_assets} × 8B)      │")
    logger.info(f"  │   Price array: {legacy_full_bytes/(1024**3):.3f} GB  "
                f"({n_paths:,} × {n_steps+1} × {n_assets} × 8B)    │")
    logger.info(f"  │   TOTAL:       {legacy_total:.3f} GB                               │")

    optimal_bs = compute_optimal_batch_size(
        n_assets=n_assets, n_steps=n_steps,
        free_vram_mb=get_gpu_memory_info().get('free_mb', 24000)
    )
    batched_rng_bytes = optimal_bs * 1 * n_assets * 8
    batched_logS_bytes = optimal_bs * n_assets * 8
    batched_result_bytes = n_paths * n_assets * 8
    batched_total = (batched_rng_bytes + batched_logS_bytes + batched_result_bytes) / (1024**3)
    logger.info(f"  │                                                             │")
    logger.info(f"  │ BATCHED (streaming):                                        │")
    logger.info(f"  │   RNG per step: {batched_rng_bytes/(1024**2):.1f} MB  "
                f"({optimal_bs:,} × 1 × {n_assets} × 8B)           │")
    logger.info(f"  │   log_S state:  {batched_logS_bytes/(1024**2):.1f} MB  "
                f"({optimal_bs:,} × {n_assets} × 8B)              │")
    logger.info(f"  │   Result ST:    {batched_result_bytes/(1024**2):.1f} MB  "
                f"({n_paths:,} × {n_assets} × 8B)          │")
    logger.info(f"  │   PEAK TOTAL:   {batched_total:.3f} GB                              │")
    logger.info(f"  │                                                             │")
    reduction = (1 - batched_total / legacy_total) * 100 if legacy_total > 0 else 0
    logger.info(f"  │ VRAM REDUCTION: {reduction:.1f}%  "
                f"({legacy_total:.3f}GB → {batched_total:.3f}GB)                │")
    logger.info(f"  └─────────────────────────────────────────────────────────────┘")


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


def run_american_option_pde(vs=None):
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE 4: American Option PDE Pricing — Crank-Nicolson + Free Boundary")
    logger.info("=" * 70)

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("")
    logger.info("  ── Part A: Constant-Vol American Put ──")
    comparison = compare_american_european(S0, K, T, r, sigma, 'put', n_S=400, n_t=1000)

    amer_result = comparison['american']
    euro_result = comparison['european_pde']

    fig_boundary = plot_early_exercise_boundary(
        amer_result, K, S0, 'put',
        comparison_results=comparison,
        title="American Put — Free Boundary (Constant σ=20%)",
    )
    fig_boundary.write_html(os.path.join(output_dir, 'american_free_boundary.html'))
    logger.info("Saved: american_free_boundary.html")

    fig_profile = plot_pde_price_profile(
        amer_result, euro_result, K, S0, 'put',
        title="PDE Price Profile — American vs European Put (σ=20%)",
    )
    fig_profile.write_html(os.path.join(output_dir, 'american_vs_european_profile.html'))
    logger.info("Saved: american_vs_european_profile.html")

    logger.info("")
    logger.info("  ── Part B: Multi-Strike American Put Scan ──")
    logger.info(f"  {'K':>8s}  {'American':>12s}  {'European':>12s}  {'Premium':>10s}  {'% Prem':>8s}  {'S* at t=0':>10s}")
    logger.info("  " + "-" * 66)

    for K_test in [90, 95, 100, 105, 110]:
        comp = compare_american_european(S0, K_test, T, r, sigma, 'put', n_S=300, n_t=800)
        amer_p = comp['american']['price']
        euro_p = comp['european_bs']
        prem = comp['early_premium']
        pct = prem / amer_p * 100 if amer_p > 0 else 0

        bt, bp = comp['american']['early_exercise_boundary']
        s_star = bp[-1] if len(bp) > 0 else 0

        logger.info(f"  {K_test:>8.1f}  {amer_p:>12.6f}  {euro_p:>12.6f}  {prem:>10.6f}  {pct:>7.2f}%  {s_star:>10.2f}")

    local_vol_func = None
    if vs is not None and vs.surface_spline is not None:
        logger.info("")
        logger.info("  ── Part C: Local-Vol American Put (Dupire from IV Surface) ──")

        loc_vol_data = dupire_local_volatility(vs, S0=S0, r=r, n_strikes=50, n_expiries=30)

        if loc_vol_data is not None and loc_vol_data['loc_vol_spline'] is not None:
            lv_spline = loc_vol_data['loc_vol_spline']

            def local_vol_func(S_val, t_val):
                try:
                    result = float(lv_spline(max(t_val, 1/365), S_val).item())
                    return np.clip(result, 0.05, 2.0)
                except Exception:
                    return sigma

            comp_lv = compare_american_european(
                S0, K, T, r, sigma, 'put',
                n_S=300, n_t=800,
                local_vol_func=local_vol_func,
            )

            amer_lv = comp_lv['american']
            fig_boundary_lv = plot_early_exercise_boundary(
                amer_lv, K, S0, 'put',
                comparison_results=comp_lv,
                title="American Put — Free Boundary (Dupire Local Vol)",
            )
            fig_boundary_lv.write_html(os.path.join(output_dir, 'american_free_boundary_localvol.html'))
            logger.info("Saved: american_free_boundary_localvol.html")
        else:
            logger.warning("Local volatility extraction failed, skipping Part C")

    logger.info("-" * 70)
    return amer_result


def main():
    logger.info("╔══════════════════════════════════════════════════════════════════════╗")
    logger.info("║   Options Pricing & Volatility Surface Analysis Platform            ║")
    logger.info("║   GPU-Batched MC │ PDE American │ Newton-Raphson IV │ 3D Visualization ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════╝")
    logger.info("")

    run_gpu_memory_diagnostics()
    mc_call, mc_put, bs_call, bs_put = run_monte_carlo_pricing()
    basket_result = run_basket_option_pricing()
    recovered_iv = run_implied_volatility_solver()
    vs = run_volatility_surface()
    amer_result = run_american_option_pde(vs)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  ALL PHASES COMPLETE — Output files saved to ./output/")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Generated Files:")
    logger.info("    📊 volatility_surface_3d.html              — 3D IV surface")
    logger.info("    📊 volatility_smile_slices.html            — IV smile slices")
    logger.info("    📊 american_free_boundary.html             — American put free boundary")
    logger.info("    📊 american_vs_european_profile.html       — PDE price profile")
    logger.info("    📊 american_free_boundary_localvol.html    — Free boundary (Dupire LV)")
    logger.info("")
    logger.info("  Architecture Highlights:")
    logger.info("    ✅ Batched streaming GPU kernel — no OOM on large baskets")
    logger.info("    ✅ Step-by-step RNG consumption — constant VRAM per time step")
    logger.info("    ✅ Auto batch-size detection — adapts to available GPU memory")
    logger.info("    ✅ Multi-asset basket options — Cholesky-correlated GBM paths")
    logger.info("    ✅ Crank-Nicolson PDE solver — American option free boundary")
    logger.info("    ✅ Dupire local volatility — from implied vol surface to LV grid")


if __name__ == '__main__':
    main()
