import numpy as np
import time
import logging
import math

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available — GPU acceleration disabled, falling back to CPU")


def get_gpu_memory_info():
    if not GPU_AVAILABLE:
        return {'total_mb': 0, 'free_mb': 0, 'used_mb': 0}
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = cp.cuda.Device(0).mem_info[1]
        free_bytes = cp.cuda.Device(0).mem_info[0]
        return {
            'total_mb': total_bytes / (1024 ** 2),
            'free_mb': free_bytes / (1024 ** 2),
            'used_mb': used_bytes / (1024 ** 2),
        }
    except Exception:
        return {'total_mb': 0, 'free_mb': 0, 'used_mb': 0}


def compute_optimal_batch_size(n_assets, n_steps, free_vram_mb=None,
                                safety_factor=0.6, element_bytes=8):
    BYTES_PER_MB = 1024 ** 2

    if free_vram_mb is None:
        info = get_gpu_memory_info()
        free_vram_mb = info['free_mb']

    usable_bytes = free_vram_mb * BYTES_PER_MB * safety_factor

    per_path_per_step = n_assets * element_bytes
    step_working_set_per_path = per_path_per_step * 3

    max_paths_for_step = int(usable_bytes / step_working_set_per_path)

    BATCH_FLOOR = 8192
    BATCH_CEIL = 262144

    batch_size = max(BATCH_FLOOR, min(max_paths_for_step, BATCH_CEIL))
    batch_size = int(2 ** math.floor(math.log2(batch_size)))

    logger.info(
        f"Auto batch size: {batch_size:,} paths/batch "
        f"(free VRAM={free_vram_mb:.0f}MB, n_assets={n_assets}, n_steps={n_steps})"
    )
    return batch_size


def _gbm_batched_gpu(S0, mu, sigma, T, n_steps, n_paths,
                      rng_seed=None, batch_size=None):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy/GPU not available")

    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol = sigma * cp.sqrt(dt)

    if batch_size is None:
        batch_size = compute_optimal_batch_size(
            n_assets=1, n_steps=n_steps, free_vram_mb=None
        )

    if rng_seed is not None:
        cp.random.seed(rng_seed)

    all_ST = cp.empty(n_paths, dtype=cp.float64)
    n_batches = math.ceil(n_paths / batch_size)
    paths_processed = 0

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_paths)
        current_batch = b_end - b_start

        log_S = cp.zeros(current_batch, dtype=cp.float64)

        for step in range(n_steps):
            Z = cp.random.standard_normal(current_batch)
            log_S += drift + vol * Z

        all_ST[b_start:b_end] = log_S
        paths_processed += current_batch

    ST = S0 * cp.exp(all_ST)
    return ST


def _gbm_paths_gpu_legacy(S0, mu, sigma, T, n_steps, n_paths, rng_seed=None):
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


def _basket_gbm_batched_gpu(S0_vec, mu_vec, sigma_vec, corr_matrix,
                             T, n_steps, n_paths, rng_seed=None,
                             batch_size=None):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy/GPU not available")

    n_assets = len(S0_vec)
    dt = T / n_steps

    S0_cp = cp.asarray(S0_vec, dtype=cp.float64)
    mu_cp = cp.asarray(mu_vec, dtype=cp.float64)
    sigma_cp = cp.asarray(sigma_vec, dtype=cp.float64)

    drift_vec = (mu_cp - 0.5 * sigma_cp ** 2) * dt
    vol_vec = sigma_cp * cp.sqrt(dt)

    L = cp.linalg.cholesky(cp.asarray(corr_matrix, dtype=cp.float64))

    if batch_size is None:
        batch_size = compute_optimal_batch_size(
            n_assets=n_assets, n_steps=n_steps, free_vram_mb=None
        )

    if rng_seed is not None:
        cp.random.seed(rng_seed)

    all_ST = cp.empty((n_paths, n_assets), dtype=cp.float64)
    n_batches = math.ceil(n_paths / batch_size)

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_paths)
        current_batch = b_end - b_start

        log_S = cp.zeros((current_batch, n_assets), dtype=cp.float64)

        for step in range(n_steps):
            Z = cp.random.standard_normal((current_batch, n_assets), dtype=cp.float64)
            correlated_Z = Z @ L.T
            log_S += drift_vec + vol_vec * correlated_Z

        all_ST[b_start:b_end] = log_S

    ST_matrix = S0_cp * cp.exp(all_ST)
    return ST_matrix


def _basket_gbm_cpu(S0_vec, mu_vec, sigma_vec, corr_matrix,
                     T, n_steps, n_paths, rng_seed=None):
    n_assets = len(S0_vec)
    dt = T / n_steps

    S0_np = np.asarray(S0_vec, dtype=np.float64)
    mu_np = np.asarray(mu_vec, dtype=np.float64)
    sigma_np = np.asarray(sigma_vec, dtype=np.float64)

    drift_vec = (mu_np - 0.5 * sigma_np ** 2) * dt
    vol_vec = sigma_np * np.sqrt(dt)

    L = np.linalg.cholesky(corr_matrix)

    rng = np.random.default_rng(rng_seed)

    log_S = np.zeros((n_paths, n_assets), dtype=np.float64)

    for step in range(n_steps):
        Z = rng.standard_normal((n_paths, n_assets))
        correlated_Z = Z @ L.T
        log_S += drift_vec + vol_vec * correlated_Z

    ST_matrix = S0_np * np.exp(log_S)
    return ST_matrix


def monte_carlo_european_gpu(S0, K, T, r, sigma, n_paths=1_000_000,
                              n_steps=252, option_type='call',
                              rng_seed=None, antithetic=True,
                              batch_size=None):
    if not GPU_AVAILABLE:
        logger.info("GPU unavailable, redirecting to CPU kernel")
        return monte_carlo_european_cpu(
            S0, K, T, r, sigma, n_paths, n_steps, option_type, rng_seed, antithetic
        )

    t0 = time.perf_counter()

    effective_paths = n_paths // 2 if antithetic else n_paths
    effective_paths = max(effective_paths, 1)

    ST = _gbm_batched_gpu(S0, r, sigma, T, n_steps, effective_paths, rng_seed, batch_size)

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

    mem_info = get_gpu_memory_info()
    elapsed = time.perf_counter() - t0

    logger.info(
        f"[GPU-BATCHED] MC {option_type.upper()} | S0={S0} K={K} T={T} σ={sigma} | "
        f"Paths={n_paths:,} | Price={price_gpu:.6f} | StdErr={std_err_gpu:.6f} | "
        f"Time={elapsed*1000:.2f}ms | VRAM: {mem_info['used_mb']:.0f}/{mem_info['total_mb']:.0f}MB"
    )

    return {
        'price': price_gpu,
        'std_error': std_err_gpu,
        'elapsed_ms': elapsed * 1000,
        'n_paths': n_paths,
        'device': 'GPU',
        'vram_used_mb': mem_info['used_mb'],
        'vram_total_mb': mem_info['total_mb'],
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
    }


def monte_carlo_basket_gpu(S0_vec, weights, K, T, r, sigma_vec, corr_matrix,
                            n_paths=1_000_000, n_steps=252, option_type='call',
                            rng_seed=None, antithetic=True, batch_size=None):
    if not GPU_AVAILABLE:
        logger.info("GPU unavailable, redirecting to CPU basket kernel")
        return monte_carlo_basket_cpu(
            S0_vec, weights, K, T, r, sigma_vec, corr_matrix,
            n_paths, n_steps, option_type, rng_seed, antithetic
        )

    t0 = time.perf_counter()

    n_assets = len(S0_vec)
    mu_vec = [r] * n_assets

    effective_paths = n_paths // 2 if antithetic else n_paths
    effective_paths = max(effective_paths, 1)

    ST_matrix = _basket_gbm_batched_gpu(
        S0_vec, mu_vec, sigma_vec, corr_matrix,
        T, n_steps, effective_paths, rng_seed, batch_size
    )

    weights_cp = cp.asarray(weights, dtype=cp.float64)
    basket_ST = ST_matrix @ weights_cp

    if option_type.lower() == 'call':
        payoff_pos = cp.maximum(basket_ST - K, 0)
    else:
        payoff_pos = cp.maximum(K - basket_ST, 0)

    if antithetic:
        sigma_cp = cp.asarray(sigma_vec, dtype=cp.float64)
        S0_cp = cp.asarray(S0_vec, dtype=cp.float64)
        log_ST_anti = 2 * (cp.asarray(mu_vec) - 0.5 * sigma_cp ** 2) * T - cp.log(ST_matrix / S0_cp)
        ST_anti = S0_cp * cp.exp(log_ST_anti)
        basket_ST_anti = ST_anti @ weights_cp
        if option_type.lower() == 'call':
            payoff_neg = cp.maximum(basket_ST_anti - K, 0)
        else:
            payoff_neg = cp.maximum(K - basket_ST_anti, 0)
        payoff = 0.5 * (payoff_pos + payoff_neg)
    else:
        payoff = payoff_pos

    discount = cp.exp(-r * T)
    price_gpu = float(discount * cp.mean(payoff))
    std_err_gpu = float(discount * cp.std(payoff) / cp.sqrt(n_paths))

    mem_info = get_gpu_memory_info()
    elapsed = time.perf_counter() - t0

    logger.info(
        f"[GPU-BATCHED] BASKET {option_type.upper()} | Assets={n_assets} | "
        f"K={K} T={T} | Paths={n_paths:,} | Price={price_gpu:.6f} | "
        f"StdErr={std_err_gpu:.6f} | Time={elapsed*1000:.2f}ms | "
        f"VRAM: {mem_info['used_mb']:.0f}/{mem_info['total_mb']:.0f}MB"
    )

    return {
        'price': price_gpu,
        'std_error': std_err_gpu,
        'elapsed_ms': elapsed * 1000,
        'n_paths': n_paths,
        'n_assets': n_assets,
        'device': 'GPU',
        'vram_used_mb': mem_info['used_mb'],
        'vram_total_mb': mem_info['total_mb'],
    }


def monte_carlo_basket_cpu(S0_vec, weights, K, T, r, sigma_vec, corr_matrix,
                            n_paths=1_000_000, n_steps=252, option_type='call',
                            rng_seed=None, antithetic=True):
    t0 = time.perf_counter()

    n_assets = len(S0_vec)
    mu_vec = [r] * n_assets

    effective_paths = n_paths // 2 if antithetic else n_paths
    effective_paths = max(effective_paths, 1)

    ST_matrix = _basket_gbm_cpu(
        S0_vec, mu_vec, sigma_vec, corr_matrix,
        T, n_steps, effective_paths, rng_seed
    )

    weights_np = np.asarray(weights, dtype=np.float64)
    basket_ST = ST_matrix @ weights_np

    if option_type.lower() == 'call':
        payoff_pos = np.maximum(basket_ST - K, 0)
    else:
        payoff_pos = np.maximum(K - basket_ST, 0)

    if antithetic:
        sigma_np = np.asarray(sigma_vec, dtype=np.float64)
        S0_np = np.asarray(S0_vec, dtype=np.float64)
        log_ST_anti = 2 * (np.asarray(mu_vec) - 0.5 * sigma_np ** 2) * T - np.log(ST_matrix / S0_np)
        ST_anti = S0_np * np.exp(log_ST_anti)
        basket_ST_anti = ST_anti @ weights_np
        if option_type.lower() == 'call':
            payoff_neg = np.maximum(basket_ST_anti - K, 0)
        else:
            payoff_neg = np.maximum(K - basket_ST_anti, 0)
        payoff = 0.5 * (payoff_pos + payoff_neg)
    else:
        payoff = payoff_pos

    discount = np.exp(-r * T)
    price_cpu = float(discount * np.mean(payoff))
    std_err_cpu = float(discount * np.std(payoff) / np.sqrt(n_paths))

    elapsed = time.perf_counter() - t0

    logger.info(
        f"[CPU] BASKET {option_type.upper()} | Assets={n_assets} | "
        f"K={K} T={T} | Paths={n_paths:,} | Price={price_cpu:.6f} | "
        f"StdErr={std_err_cpu:.6f} | Time={elapsed*1000:.2f}ms"
    )

    return {
        'price': price_cpu,
        'std_error': std_err_cpu,
        'elapsed_ms': elapsed * 1000,
        'n_paths': n_paths,
        'n_assets': n_assets,
        'device': 'CPU',
    }
