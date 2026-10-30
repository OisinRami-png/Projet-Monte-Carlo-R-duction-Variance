"""Pricing d'un call européen par Monte-Carlo et méthodes de réduction de variance.

Modèle de Black-Scholes : S_T = S0 * exp((r - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0, 1).
Chaque estimateur renvoie (prix, (borne_inf, borne_sup)) avec un IC à 95 %.
"""
import numpy as np
from scipy.stats import norm

Z_95 = 1.96


def call_bs(S0, K, r, sigma, T):
    """Prix exact du call européen (formule fermée de Black-Scholes)."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def call_mc(S0, K, r, sigma, T, N=1_000_000, seed=0):
    """Monte-Carlo naïf."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    X = np.exp(-r * T) * np.maximum(ST - K, 0)
    price = X.mean()
    se = X.std(ddof=1) / np.sqrt(N)
    return price, (price - Z_95 * se, price + Z_95 * se)


def call_mc_antithetique(S0, K, r, sigma, T, N=500_000, seed=0):
    """Variables antithétiques : on simule Z et -Z (N paires = 2N trajectoires)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    ST_plus = S0 * np.exp(drift + vol * Z)
    ST_moins = S0 * np.exp(drift - vol * Z)
    X_plus = np.exp(-r * T) * np.maximum(ST_plus - K, 0)
    X_moins = np.exp(-r * T) * np.maximum(ST_moins - K, 0)
    Y = 0.5 * (X_plus + X_moins)
    price = Y.mean()
    se = Y.std(ddof=1) / np.sqrt(N)
    return price, (price - Z_95 * se, price + Z_95 * se)


def call_mc_controle(S0, K, r, sigma, T, N=1_000_000, seed=0):
    """Variable de contrôle Y = e^{-rT} S_T, d'espérance connue E[Y] = S0."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    X = np.exp(-r * T) * np.maximum(ST - K, 0)
    Y = np.exp(-r * T) * ST
    C = np.cov(X, Y)
    beta = C[0, 1] / C[1, 1]
    W = X - beta * (Y - S0)
    price = W.mean()
    se = W.std(ddof=1) / np.sqrt(N)
    return price, (price - Z_95 * se, price + Z_95 * se)


def n_minimum(sigma_X, epsilon, z=Z_95):
    """Nombre minimal de simulations pour un IC à 95 % de demi-largeur epsilon."""
    return int(np.ceil((z * sigma_X / epsilon) ** 2))


def ratios_reduction(S0, K, r, sigma, T, N=200_000, seed=0):
    """Facteurs de réduction de variance à coût égal (même nombre de payoffs évalués).

    Renvoie (Var_naif / Var_antithétique, Var_naif / Var_contrôle).
    Pour l'antithétique, une paire coûte 2 simulations : on compare Var(X) à 2 Var(Y).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    disc = np.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + vol * Z)
    X = disc * np.maximum(ST - K, 0)
    X_moins = disc * np.maximum(S0 * np.exp(drift - vol * Z) - K, 0)
    var_naif = X.var(ddof=1)
    var_anti = (0.5 * (X + X_moins)).var(ddof=1)
    Y = disc * ST
    C = np.cov(X, Y)
    var_ctrl = (X - C[0, 1] / C[1, 1] * (Y - S0)).var(ddof=1)
    return var_naif / (2 * var_anti), var_naif / var_ctrl
