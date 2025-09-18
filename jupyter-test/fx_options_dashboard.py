import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import plotly as px
from scipy.stats import norm
import streamlit as st
class FX_Options_Model:
    def __init__(self, S0, K, T, rd, rf, vol):
        self.S0 = S0      # Spot price
        self.K = K        # Strike
        self.T = T        # Time to expiry
        self.rd = rd      # Domestic interest rate
        self.rf = rf      # Foreign interest rate
        self.vol = vol    # Volatility

    def _mc_simulation(self, num_simulations, reduce_variance=True, seed=None):
        rng = np.random.default_rng(seed)
        n = num_simulations // (2 if reduce_variance else 1) ##half if we take Z and -Z, round to integer 
        Z = rng.standard_normal(n)
        if reduce_variance:
            Z = np.concatenate([Z, -Z])
        return Z
    
    def price_vanilla_call(self, num_simulations=100000, reduce_variance=True, seed=None):
        Z = self._mc_simulation(num_simulations, reduce_variance, seed)
        
        drift = (self.rd - self.rf - 0.5 * self.vol**2) * self.T
        diffusion = self.vol * np.sqrt(self.T) * Z
        S_t = self.S0 * np.exp(drift + diffusion)
        payoff = np.maximum(S_t - self.K, 0)
        call_value = np.exp(-self.rd * self.T) * payoff
        fair_price = np.mean(call_value)
        std_error = call_value.std(ddof=1) / np.sqrt(Z.size)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)
        # 1.96 = critical vlaue of 95% confidence interval
        return fair_price, std_error, conf_interval
    
    def convergence_data(self, num_points = 10):
        min_N = 4
        max_N = 7
        Ns = np.logspace(min_N, max_N, num_points, dtype=int)
        
        prices, errors = [], []
        for N in Ns:
            fair_price, std_error, _ = self.price_vanilla_call(num_simulations=N)
            prices.append(fair_price)
            errors.append(std_error)

        prices, errors = np.array(prices), np.array(errors)
        return Ns, prices, errors
    
    def plot_convergence(self, num_points = 10):
        Ns, prices, errors = self.convergence_data(num_points)
        plt.errorbar(
            Ns, prices,
            yerr=1.96 * errors,
            fmt="o-", capsize=4, label="FX option price ±95% CI"
        )
        plt.xscale('log')
        plt.xlabel('No. simulations')
        plt.ylabel('Model price')
        plt.title('FX Option fair value')
        plt.legend()
        plt.show()
        
st.title("FX Options Pricer Dashboard")

# Input boxes
S0 = st.number_input("Spot FX (S0)", value=1.10, format="%.4f")
K = st.number_input("Strike (K)", value=1.12, format="%.4f")
T = st.number_input("Expiry (T, in years)", value=1.0, format="%.2f")
rd = st.number_input("Domestic Rate (rd)", value=0.05, format="%.4f")
rf = st.number_input("Foreign Rate (rf)", value=0.03, format="%.4f")
vol = st.number_input("Volatility (vol))", value=0.10, format="%.4f")

# Run model when button is pressed
if st.button("Run model"):
    model = FX_Options_Model(S0, K, T, rd, rf, vol)

    fair_price, std_error, ci = model.price_vanilla_call(num_simulations=100000, seed=42)

    st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
    st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

    # Convergence plot
    Ns, prices, errors = model.convergence_data()
    fig, ax = plt.subplots()
    ax.errorbar(Ns, prices, yerr=1.96 * errors, fmt="o-", capsize=4, label="MC price ±95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Estimated call price")
    ax.set_title("Monte Carlo Convergence")
    ax.legend()
    st.pyplot(fig)