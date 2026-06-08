import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def black_scholes_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type.lower() == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type.lower() == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def black_scholes_vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega


def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def black_scholes_theta(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type.lower() == 'call':
        theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta / 365.0


def black_scholes_rho(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return 0.0

    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    return rho
