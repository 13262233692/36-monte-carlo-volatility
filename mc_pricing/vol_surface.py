import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RectBivariateSpline
import logging

from .implied_vol import newton_raphson_iv_vectorized

logger = logging.getLogger(__name__)


class VolatilitySurface:
    def __init__(self, option_data, r=0.05, option_type='call'):
        self.raw_data = option_data.copy()
        self.r = r
        self.option_type = option_type
        self.iv_grid = None
        self.strikes_unique = None
        self.expiries_unique = None
        self.surface_spline = None

    def compute_implied_volatilities(self):
        df = self.raw_data
        S = df['spot'].iloc[0]
        strikes = df['strike'].values.astype(float)
        expiries = df['T'].values.astype(float)
        prices = df['mid_price'].values.astype(float)

        logger.info(f"Computing IVs for {len(prices)} options (S={S:.2f}, r={self.r})")

        ivs = newton_raphson_iv_vectorized(
            market_prices=prices,
            S=S,
            strikes=strikes,
            expiries=expiries,
            r=self.r,
            option_type=self.option_type,
        )

        self.raw_data['implied_vol'] = ivs
        valid_mask = ~np.isnan(ivs) & (ivs > 0.01) & (ivs < 3.0)
        n_total = len(ivs)
        n_valid = valid_mask.sum()
        logger.info(f"IV computation: {n_valid}/{n_total} valid ({100*n_valid/n_total:.1f}%)")

        self.raw_data = self.raw_data[valid_mask].reset_index(drop=True)
        return self.raw_data

    def build_surface(self, n_strike_points=100, n_expiry_points=50):
        df = self.raw_data
        if 'implied_vol' not in df.columns:
            self.compute_implied_volatilities()

        df = self.raw_data
        self.strikes_unique = np.sort(df['strike'].unique())
        self.expiries_unique = np.sort(df['T'].unique())

        pivot = df.pivot_table(values='implied_vol', index='T', columns='strike', aggfunc='mean')
        pivot = pivot.interpolate(axis=1, limit_direction='both').interpolate(axis=0, limit_direction='both')

        T_vals = pivot.index.values.astype(float)
        K_vals = pivot.columns.values.astype(float)
        IV_matrix = pivot.values.astype(float)

        valid_T = T_vals[T_vals > 0]
        if len(valid_T) < 2:
            logger.warning("Not enough unique expiries for 2D interpolation, using 1D spline per slice")
            self._build_1d_slices(K_vals, IV_matrix, T_vals, n_strike_points)
            return self

        try:
            self.surface_spline = RectBivariateSpline(T_vals, K_vals, IV_matrix, kx=3, ky=3)
            self.iv_grid = self._evaluate_surface(n_strike_points, n_expiry_points)
            logger.info(f"Built 2D cubic spline surface: {n_strike_points}x{n_expiry_points} grid")
        except Exception as e:
            logger.warning(f"2D spline failed ({e}), falling back to 1D slices")
            self._build_1d_slices(K_vals, IV_matrix, T_vals, n_strike_points)

        return self

    def _build_1d_slices(self, K_vals, IV_matrix, T_vals, n_points):
        strike_grid = np.linspace(K_vals.min(), K_vals.max(), n_points)
        all_ivs = []

        for i, T in enumerate(T_vals):
            if T <= 0:
                continue
            iv_slice = IV_matrix[i, :]
            valid = ~np.isnan(iv_slice) & (iv_slice > 0)
            if valid.sum() < 4:
                continue
            cs = CubicSpline(K_vals[valid], iv_slice[valid], bc_type='natural')
            iv_interp = cs(strike_grid)
            iv_interp = np.clip(iv_interp, 0.01, 3.0)
            all_ivs.append({'T': T, 'iv': iv_interp})

        if not all_ivs:
            logger.error("Could not build any IV slices")
            return

        expiry_grid = np.array([x['T'] for x in all_ivs])
        iv_array = np.array([x['iv'] for x in all_ivs])

        from scipy.interpolate import RectBivariateSpline
        try:
            self.surface_spline = RectBivariateSpline(expiry_grid, strike_grid, iv_array, kx=3, ky=3)
        except Exception:
            self.surface_spline = None

        self.iv_grid = {
            'strikes': strike_grid,
            'expiries': expiry_grid,
            'iv_surface': iv_array,
        }

    def _evaluate_surface(self, n_k, n_t):
        K_min, K_max = self.strikes_unique.min(), self.strikes_unique.max()
        T_min, T_max = self.expiries_unique.min(), self.expiries_unique.max()

        strike_grid = np.linspace(K_min, K_max, n_k)
        expiry_grid = np.linspace(max(T_min, 1/365), T_max, n_t)

        iv_surface = self.surface_spline(expiry_grid, strike_grid)
        iv_surface = np.clip(iv_surface, 0.01, 3.0)

        return {
            'strikes': strike_grid,
            'expiries': expiry_grid,
            'iv_surface': iv_surface,
        }

    def get_iv(self, T, K):
        if self.surface_spline is None:
            return np.nan
        return float(self.surface_spline(T, K))

    def get_surface_arrays(self):
        if self.iv_grid is None:
            raise ValueError("Surface not yet built — call build_surface() first")
        return self.iv_grid
