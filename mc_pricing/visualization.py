import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_volatility_surface_3d(surface_data, title="Implied Volatility Surface — Volatility Smile",
                                colorscale='Viridis', show_scatter=True, scatter_data=None):
    strikes = surface_data['strikes']
    expiries = surface_data['expiries']
    iv_surface = surface_data['iv_surface']

    K_mesh, T_mesh = np.meshgrid(strikes, expiries)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=K_mesh,
        y=T_mesh * 365,
        z=iv_surface * 100,
        colorscale=colorscale,
        opacity=0.92,
        colorbar=dict(
            title=dict(text='IV (%)', font=dict(size=14)),
            tickfont=dict(size=12),
            len=0.75,
        ),
        lighting=dict(
            ambient=0.4,
            diffuse=0.6,
            specular=0.2,
            roughness=0.5,
            fresnel=0.3,
        ),
        lightposition=dict(
            x=0,
            y=-1000,
            z=2000,
        ),
        hovertemplate='Strike: %{x:.1f}<br>Expiry: %{y:.0f} days<br>IV: %{z:.2f}%<extra></extra>',
        name='IV Surface',
    ))

    if show_scatter and scatter_data is not None:
        df = scatter_data
        if 'implied_vol' in df.columns:
            valid = df.dropna(subset=['implied_vol'])
            fig.add_trace(go.Scatter3d(
                x=valid['strike'],
                y=valid['T'] * 365,
                z=valid['implied_vol'] * 100,
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    opacity=0.6,
                    symbol='circle',
                ),
                name='Market IV Points',
                hovertemplate='Strike: %{x:.1f}<br>Expiry: %{y:.0f} days<br>IV: %{z:.2f}%<extra></extra>',
            ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, color='white'),
            x=0.5,
            xanchor='center',
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='Strike Price (K)', font=dict(size=14, color='white')),
                gridcolor='rgba(128,128,128,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,20,40,0.8)',
                tickfont=dict(size=11, color='lightgray'),
            ),
            yaxis=dict(
                title=dict(text='Time to Expiry (Days)', font=dict(size=14, color='white')),
                gridcolor='rgba(128,128,128,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,20,40,0.8)',
                tickfont=dict(size=11, color='lightgray'),
            ),
            zaxis=dict(
                title=dict(text='Implied Volatility (%)', font=dict(size=14, color='white')),
                gridcolor='rgba(128,128,128,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,20,40,0.8)',
                tickfont=dict(size=11, color='lightgray'),
            ),
            camera=dict(
                eye=dict(x=1.8, y=-1.5, z=1.2),
                center=dict(x=0, y=0, z=-0.15),
            ),
            bgcolor='rgba(10,10,30,0.95)',
        ),
        paper_bgcolor='rgba(10,10,30,0.95)',
        plot_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white'),
        width=1200,
        height=800,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            font=dict(size=12, color='white'),
            bgcolor='rgba(30,30,60,0.8)',
            bordercolor='gray',
            borderwidth=1,
        ),
    )

    return fig


def plot_mc_convergence(mc_results, bs_price, title="Monte Carlo Convergence vs Black-Scholes"):
    fig = go.Figure()

    fig.add_hline(
        y=bs_price,
        line_dash="dash",
        line_color="#00ff88",
        line_width=2,
        annotation_text=f"Black-Scholes: {bs_price:.4f}",
        annotation_position="top left",
        annotation_font=dict(size=13, color="#00ff88"),
    )

    fig.add_hrect(
        y0=bs_price - 2 * mc_results['std_error'],
        y1=bs_price + 2 * mc_results['std_error'],
        fillcolor="rgba(0,255,136,0.08)",
        line_width=0,
    )

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=mc_results['price'],
        delta=dict(
            reference=bs_price,
            relative=True,
            valueformat='.4f',
            increasing_color='#ff4444',
            decreasing_color='#44ff44',
        ),
        title=dict(text="MC Price", font=dict(size=16, color='white')),
        number=dict(font=dict(size=28, color='#00ccff')),
        domain=dict(x=[0.6, 0.85], y=[0.7, 0.95]),
    ))

    stats_text = (
        f"<b>Monte Carlo Results</b><br>"
        f"Paths: {mc_results['n_paths']:,}<br>"
        f"Device: {mc_results['device']}<br>"
        f"Price: {mc_results['price']:.6f}<br>"
        f"Std Error: {mc_results['std_error']:.6f}<br>"
        f"Elapsed: {mc_results['elapsed_ms']:.2f} ms<br>"
        f"<br><b>Black-Scholes</b><br>"
        f"Price: {bs_price:.6f}<br>"
        f"<br><b>Error</b><br>"
        f"Absolute: {abs(mc_results['price'] - bs_price):.6f}<br>"
        f"Relative: {abs(mc_results['price'] - bs_price) / bs_price * 100:.4f}%"
    )

    fig.add_annotation(
        x=0.02,
        y=0.5,
        text=stats_text,
        showarrow=False,
        font=dict(size=13, color='white', family='Consolas'),
        align='left',
        bordercolor='gray',
        borderwidth=1,
        borderpad=10,
        bgcolor='rgba(20,20,50,0.9)',
        xref='paper',
        yref='paper',
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
        ),
        paper_bgcolor='rgba(10,10,30,0.95)',
        plot_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white'),
        width=1200,
        height=600,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
    )

    return fig


def plot_price_distribution(mc_results, S0, K, T, r, sigma, option_type='call',
                            title="Terminal Price Distribution & Option Payoff"):
    paths = mc_results['paths_matrix']
    try:
        import cupy as cp
        if hasattr(paths, 'get'):
            ST = cp.asnumpy(paths[:, -1])
        else:
            ST = np.asarray(paths[:, -1])
    except ImportError:
        ST = np.asarray(paths[:, -1])

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=('Terminal Stock Price Distribution', 'Option Payoff Distribution'),
    )

    hist_ST = np.histogram(ST, bins=80, density=True)
    fig.add_trace(go.Bar(
        x=hist_ST[1][:-1],
        y=hist_ST[0],
        width=np.diff(hist_ST[1]),
        marker_color='rgba(0,150,255,0.6)',
        marker_line=dict(color='rgba(0,200,255,0.8)', width=0.5),
        name='S_T Distribution',
    ), row=1, col=1)

    fig.add_vline(
        x=K,
        line_dash="dash",
        line_color="#ff6644",
        line_width=2,
        annotation_text=f"K={K}",
        row=1, col=1,
    )

    if option_type.lower() == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discounted = payoffs * np.exp(-r * T)

    hist_payoff = np.histogram(discounted[discounted > 0], bins=60, density=True)
    fig.add_trace(go.Bar(
        x=hist_payoff[1][:-1],
        y=hist_payoff[0],
        width=np.diff(hist_payoff[1]),
        marker_color='rgba(255,100,50,0.6)',
        marker_line=dict(color='rgba(255,150,80,0.8)', width=0.5),
        name='Discounted Payoff (ITM)',
    ), row=2, col=1)

    mc_price = mc_results['price']
    fig.add_vline(
        x=mc_price,
        line_dash="dot",
        line_color="#00ff88",
        line_width=2,
        annotation_text=f"MC Price={mc_price:.4f}",
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
        ),
        paper_bgcolor='rgba(10,10,30,0.95)',
        plot_bgcolor='rgba(10,10,40,0.9)',
        font=dict(color='white'),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(font=dict(size=12, color='white'), bgcolor='rgba(20,20,50,0.8)'),
    )

    fig.update_xaxes(
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(color='lightgray'),
        title_font=dict(color='white'),
        row=1, col=1,
    )
    fig.update_xaxes(
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(color='lightgray'),
        title_font=dict(color='white'),
        row=2, col=1,
    )
    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(color='lightgray'),
        title_font=dict(color='white'),
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor='rgba(128,128,128,0.3)',
        tickfont=dict(color='lightgray'),
        title_font=dict(color='white'),
        row=2, col=1,
    )

    return fig


def plot_volatility_smile_slices(surface_data, slices_at_days=[30, 90, 180, 365],
                                  title="Volatility Smile Slices"):
    strikes = surface_data['strikes']
    expiries = surface_data['expiries']
    iv_surface = surface_data['iv_surface']

    colors = ['#ff4444', '#ffaa00', '#44ff44', '#00ccff', '#aa44ff', '#ff44aa']

    fig = go.Figure()

    for i, target_day in enumerate(slices_at_days):
        target_T = target_day / 365.0
        idx = np.argmin(np.abs(expiries - target_T))
        actual_day = expiries[idx] * 365

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=strikes,
            y=iv_surface[idx, :] * 100,
            mode='lines+markers',
            name=f'T = {actual_day:.0f} days',
            line=dict(color=color, width=2.5),
            marker=dict(size=4, color=color),
            hovertemplate='Strike: %{x:.1f}<br>IV: %{y:.2f}%<extra></extra>',
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text='Strike Price (K)', font=dict(size=14, color='white')),
            gridcolor='rgba(128,128,128,0.3)',
            tickfont=dict(color='lightgray'),
        ),
        yaxis=dict(
            title=dict(text='Implied Volatility (%)', font=dict(size=14, color='white')),
            gridcolor='rgba(128,128,128,0.3)',
            tickfont=dict(color='lightgray'),
        ),
        paper_bgcolor='rgba(10,10,30,0.95)',
        plot_bgcolor='rgba(10,10,40,0.9)',
        font=dict(color='white'),
        width=1200,
        height=600,
        legend=dict(font=dict(size=12, color='white'), bgcolor='rgba(20,20,50,0.8)'),
    )

    return fig
