import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not available — synthetic market data will be used")


def fetch_option_chain(ticker='SPY', max_expiries=6):
    if not YF_AVAILABLE:
        logger.info("yfinance unavailable, generating synthetic data")
        return generate_synthetic_market_data()

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        if hist.empty:
            raise ValueError(f"No price history for {ticker}")
        spot = hist['Close'].iloc[-1]
        logger.info(f"Fetched spot price for {ticker}: {spot:.2f}")

        expiries_list = stock.options
        if not expiries_list:
            raise ValueError(f"No options data for {ticker}")

        expiries = expiries_list[:max_expiries]
        all_data = []

        for exp_date_str in expiries:
            try:
                chain = stock.option_chain(exp_date_str)
            except Exception as e:
                logger.warning(f"Failed to fetch chain for {exp_date_str}: {e}")
                continue

            calls = chain.calls
            if calls.empty:
                continue

            exp_date = pd.Timestamp(exp_date_str)
            T = (exp_date - pd.Timestamp.now()).days / 365.0

            if T <= 0.01:
                continue

            for _, row in calls.iterrows():
                K = row['strike']
                bid = row.get('bid', 0)
                ask = row.get('ask', 0)

                if bid <= 0 and ask <= 0:
                    continue

                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                elif ask > 0:
                    mid = ask * 0.95
                else:
                    mid = bid * 1.05

                if mid <= 0.05:
                    continue

                moneyness = K / spot
                if moneyness < 0.8 or moneyness > 1.3:
                    continue

                all_data.append({
                    'ticker': ticker,
                    'spot': spot,
                    'strike': K,
                    'expiry': exp_date_str,
                    'T': T,
                    'moneyness': moneyness,
                    'mid_price': mid,
                    'bid': bid,
                    'ask': ask,
                    'volume': row.get('volume', 0) if pd.notna(row.get('volume', 0)) else 0,
                    'openInterest': row.get('openInterest', 0) if pd.notna(row.get('openInterest', 0)) else 0,
                    'option_type': 'call',
                })

        if not all_data:
            logger.warning("No valid option data fetched, falling back to synthetic")
            return generate_synthetic_market_data()

        df = pd.DataFrame(all_data)
        logger.info(f"Fetched {len(df)} option quotes for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return generate_synthetic_market_data()


def generate_synthetic_market_data(S0=100.0, r=0.05, base_sigma=0.20):
    np.random.seed(42)

    strikes = np.arange(75, 135, 2.5)
    expiries_days = [30, 60, 90, 120, 180, 270, 365]
    expiries_years = [d / 365.0 for d in expiries_days]

    rows = []
    for T in expiries_years:
        for K in strikes:
            moneyness = K / S0
            skew = 0.12 * (moneyness - 1.0) ** 2
            term_structure = 0.015 * np.sqrt(T)
            sigma = base_sigma + skew + term_structure + np.random.normal(0, 0.003)
            sigma = max(sigma, 0.05)

            from .black_scholes import black_scholes_price
            price = black_scholes_price(S0, K, T, r, sigma, 'call')
            noise = np.random.normal(0, price * 0.003)
            market_price = max(price + noise, 0.01)

            rows.append({
                'ticker': 'SYNTHETIC',
                'spot': S0,
                'strike': K,
                'expiry': f"{int(T*365)}D",
                'T': T,
                'moneyness': moneyness,
                'mid_price': market_price,
                'bid': max(market_price * 0.98, 0.01),
                'ask': market_price * 1.02,
                'volume': int(np.random.exponential(500)),
                'openInterest': int(np.random.exponential(1000)),
                'option_type': 'call',
                'true_sigma': sigma,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} synthetic option quotes (S0={S0}, base_σ={base_sigma})")
    return df
