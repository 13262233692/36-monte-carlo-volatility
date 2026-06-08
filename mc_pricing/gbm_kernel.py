import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available — GPU acceleration disabled, falling back to CPU")


def _gbm_paths_gpu(S0, mu, sigma, T, n_steps, n_paths, rng_seed=None):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy/GPU not available")

    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * cp.sqrt(dt)

    if rng_seed is not None:
        cp.random.seed(rng_seed)

    Z = cp.random.standard_normal((n_paths, n_steps))
    log_increments = drift + vol * Z
    log_prices = cp.cumsum(log_increments, axis=1)
    log_prices = cp.concatenate(
        [cp.zeros((n_paths, 1)), log_prices], axis=1
    )
    prices = S0 * cp.exp(log_prices)
    return prices


def _gbm_paths_cpu(S0, mu, sigma, T, n_steps, n_paths, rng_seed=None):
    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    rng = np.random.default_rng(rng_seed)
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = drift + vol * Z
    log_prices = np.cumsum(log_increments, axis=1)
    log_prices = np.concatenate(
        [np.zeros((n_paths, 1)), log_prices], axis=1
    )
    prices = S0 * np.exp(log_prices)
    return prices


def monte_carlo_european_gpu(S0, K, T, r, sigma, n_paths=1_000_000,
                              n_steps=252, option_type='call',
                              rng_seed=None, antithetic=True):
    if not GPU_AVAILABLE:
        logger.info("GPU unavailable, redirecting to CPU kernel")
        return monte_carlo_european_cpu(
            S0, K, T, r, sigma, n_paths, n_steps, option_type, rng_seed, antithetic
        )

    t0 = time.perf_counter()

    effective_paths = n_paths // 2 if antithetic else n_paths
    effective_paths = max(effective_paths, 1)

    prices = _gbm_paths_gpu(S0, r, sigma, T, n_steps, effective_paths, rng_seed)
    ST = prices[:, -1]

    if option_type.lower() == 'call':
        payoff_pos = cp.maximum(ST - K, 0)
    else:
        payoff_pos = cp.maximum(K - ST, 0)

    if antithetic:
        ST_anti = S0 * cp.exp(2 * (r - 0.5 * sigma ** 2) * T - (cp.log(ST / S0)))
        if option_type.lower() == 'call':
            payoff_neg = cp.maximum(ST_anti - K, 0)
        else:
            payoff_neg = cp.maximum(K - ST_anti, 0)
        payoff = 0.5 * (payoff_pos + payoff_neg)
    else:
        payoff = payoff_pos

    discount = cp.exp(-r * T)
    price_gpu = float(discount * cp.mean(payoff))
    std_err_gpu = float(discount * cp.std(payoff) / cp.sqrt(n_paths))

    elapsed = time.perf_counter() - t0

    logger.info(
        f"[GPU] MC {option_type.upper()} | S0={S0} K={K} T={T} σ={sigma} | "
        f"Paths={n_paths:,} | Price={price_gpu:.6f} | StdErr={std_err_gpu:.6f} | "
        f"Time={elapsed*1000:.2f}ms"
    )

    return {
        'price': price_gpu,
        'std_error': std_err_gpu,
        'elapsed_ms': elapsed * 1000,
        'n_paths': n_paths,
        'device': 'GPU',
        'paths_matrix': prices,
    }


def monte_carlo_european_cpu(S0, K, T, r, sigma, n_paths=1_000_000,
                              n_steps=252, option_type='call',
                              rng_seed=None, antithetic=True):
    t0 = time.perf_counter()

    effective_paths = n_paths // 2 if antithetic else n_paths
    effective_paths = max(effective_paths, 1)

    prices = _gbm_paths_cpu(S0, r, sigma, T, n_steps, effective_paths, rng_seed)
    ST = prices[:, -1]

    if option_type.lower() == 'call':
        payoff_pos = np.maximum(ST - K, 0)
    else:
        payoff_pos = np.maximum(K - ST, 0)

    if antithetic:
        ST_anti = S0 * np.exp(2 * (r - 0.5 * sigma ** 2) * T - (np.log(ST / S0)))
        if option_type.lower() == 'call':
            payoff_neg = np.maximum(ST_anti - K, 0)
        else:
            payoff_neg = np.maximum(K - ST_anti, 0)
        payoff = 0.5 * (payoff_pos + payoff_neg)
    else:
        payoff = payoff_pos

    discount = np.exp(-r * T)
    price_cpu = float(discount * np.mean(payoff))
    std_err_cpu = float(discount * np.std(payoff) / np.sqrt(n_paths))

    elapsed = time.perf_counter() - t0

    logger.info(
        f"[CPU] MC {option_type.upper()} | S0={S0} K={K} T={T} σ={sigma} | "
        f"Paths={n_paths:,} | Price={price_cpu:.6f} | StdErr={std_err_cpu:.6f} | "
        f"Time={elapsed*1000:.2f}ms"
    )

    return {
        'price': price_cpu,
        'std_error': std_err_cpu,
        'elapsed_ms': elapsed * 1000,
        'n_paths': n_paths,
        'device': 'CPU',
        'paths_matrix': prices,
    }
