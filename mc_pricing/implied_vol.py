import numpy as np
import logging
from .black_scholes import black_scholes_price, black_scholes_vega

logger = logging.getLogger(__name__)

IV_DEFAULT_INIT = 0.25
IV_MAX_ITER = 100
IV_TOL = 1e-8
IV_MIN_SIGMA = 1e-6
IV_MAX_SIGMA = 5.0


def newton_raphson_iv(market_price, S, K, T, r, option_type='call',
                       initial_sigma=None, max_iter=IV_MAX_ITER,
                       tol=IV_TOL):
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        if market_price <= intrinsic + 1e-8:
            return 0.0
        return np.nan

    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price < intrinsic - 1e-8:
        logger.warning(f"Market price {market_price:.4f} below intrinsic {intrinsic:.4f}")
        return np.nan

    sigma = initial_sigma if initial_sigma is not None else _initial_guess(market_price, S, K, T, r, option_type)
    sigma = np.clip(sigma, 0.01, IV_MAX_SIGMA)

    best_sigma = sigma
    best_diff = abs(black_scholes_price(S, K, T, r, sigma, option_type) - market_price)

    for i in range(max_iter):
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = black_scholes_vega(S, K, T, r, sigma)

        diff = bs_price - market_price
        abs_diff = abs(diff)

        if abs_diff < best_diff:
            best_diff = abs_diff
            best_sigma = sigma

        if abs_diff < tol:
            return sigma

        if abs(vega) < 1e-12:
            sigma = sigma * 1.5
            sigma = min(sigma, IV_MAX_SIGMA)
            if sigma >= IV_MAX_SIGMA:
                break
            continue

        step = diff / vega
        step = np.clip(step, -2.0 * sigma, 2.0 * sigma)

        sigma_new = sigma - step
        sigma_new = np.clip(sigma_new, 0.005, IV_MAX_SIGMA)

        if abs(sigma_new - sigma) < tol:
            return sigma_new

        sigma = sigma_new

    if best_diff < abs(black_scholes_price(S, K, T, r, sigma, option_type) - market_price):
        sigma = best_sigma

    if best_diff < market_price * 0.05:
        return sigma

    logger.warning(f"Newton-Raphson did not converge (best_diff={best_diff:.8f}, sigma={sigma:.8f})")
    return sigma


def _initial_guess(market_price, S, K, T, r, option_type):
    moneyness = S / K

    atm_approx = np.sqrt(2 * np.pi / T) * market_price / S
    atm_approx = np.clip(atm_approx, 0.05, 2.0)

    if abs(moneyness - 1.0) < 0.05:
        return atm_approx

    if option_type == 'call' and moneyness < 0.9:
        guess = max(atm_approx, 0.25 + 0.3 * (1.0 - moneyness))
    elif option_type == 'put' and moneyness > 1.1:
        guess = max(atm_approx, 0.25 + 0.3 * (moneyness - 1.0))
    else:
        guess = max(atm_approx, 0.25)

    return np.clip(guess, 0.05, 2.0)


def newton_raphson_iv_vectorized(market_prices, S, strikes, expiries, r,
                                  option_type='call', initial_sigma=None,
                                  max_iter=IV_MAX_ITER, tol=IV_TOL):
    n = len(market_prices)
    ivs = np.full(n, np.nan)

    for i in range(n):
        try:
            ivs[i] = newton_raphson_iv(
                market_price=market_prices[i],
                S=S,
                K=strikes[i],
                T=expiries[i],
                r=r,
                option_type=option_type,
                initial_sigma=initial_sigma,
                max_iter=max_iter,
                tol=tol,
            )
        except Exception as e:
            logger.warning(f"IV solve failed for K={strikes[i]}, T={expiries[i]}: {e}")
            ivs[i] = np.nan

    return ivs
