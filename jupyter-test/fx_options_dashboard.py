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

    def _mc_simulation(self, num_simulations, n_steps = 1, reduce_variance=True, seed=None):
        rng = np.random.default_rng(seed)
        n = num_simulations // (2 if reduce_variance else 1) ##half if we take Z and -Z, round to integer 
        Z = rng.standard_normal((n, n_steps))
        if reduce_variance:
            Z = np.vstack([Z, -Z]) #concatenate arrays
        return Z
    
    def price_vanilla_call(self, num_simulations=100000, reduce_variance=True, seed=None):
        Z = self._mc_simulation(num_simulations=num_simulations,n_steps=1,
        reduce_variance=reduce_variance, seed=seed).flatten()
        
        drift = (self.rd - self.rf - 0.5 * self.vol**2) * self.T
        sigma = self.vol * np.sqrt(self.T) * Z
        S_t = self.S0 * np.exp(drift + sigma)
        payoff = np.maximum(S_t - self.K, 0)
        call_value = np.exp(-self.rd * self.T) * payoff
        fair_price = np.mean(call_value)
        std_error = call_value.std(ddof=1) / np.sqrt(Z.size)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)
        # 1.96 = critical vlaue of 95% confidence interval
        return fair_price, std_error, conf_interval
    
    def price_knockout_call(self, barrier, type, num_simulations = 10000, n_steps = 252, reduce_variance = True, seed = None):
        rng = np.random.default_rng(seed)
        #need type: 'down and out' or 'up and out', and a barrier
        dt = self.T / n_steps
        drift = (self.rd - self.rf - 0.5 * self.vol**2) * dt
        sigma = self.vol * np.sqrt(dt)

        S = np.full(num_simulations, self.S0, dtype = float)
        alive = np.ones(num_simulations, dtype=bool)

        logB = np.log(barrier)
        sig2dt = (self.vol**2) * dt

        for _ in range(n_steps):
            S_old = S.copy()
            Z = rng.standard_normal(num_simulations)
            S *= np.exp(drift + sigma * Z)
            if type == "down-and-out":
                alive &= (S > barrier)
            elif type == "up-and-out":
                alive &= (S < barrier)
            else:
                raise ValueError("type must be 'down-and-out' or 'up-and-out'")
            
        # Brownian bridge correction    
            idx = np.where(alive)[0]
            if idx.size:
                x0 = np.log(S_old[idx])
                x1 = np.log(S[idx])

                if type == "down-and-out":
                    m = np.minimum(x0, x1)
                    p = np.exp(-2.0 * (x0 - logB) * (x1 - logB) / sig2dt)
                    p = np.where(logB <= m, p, 0.0)
                elif type == "up-and-out":
                    m = np.maximum(x0, x1)
                    p = np.exp(-2.0 * (logB - x0) * (logB - x1) / sig2dt)
                    p = np.where(logB >= m, p, 0.0)
                else:
                    raise ValueError("type must be 'down-and-out' or 'up-and-out'")
                
                u = rng.random(idx.size)
                knockout = (u < p)
                if np.any(knockout):
                    alive[idx[knockout]] = False

        payoff = np.where(alive, np.maximum(S - self.K, 0), 0)
        discounted = np.exp(-self.rd * self.T) * payoff

        fair_price = np.mean(discounted)
        std_error = discounted.std(ddof=1) / np.sqrt(num_simulations)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)

        return fair_price, std_error, conf_interval
    
    def convergence_data(self, option_type, num_points = 8, barrier = None, type = None, n_steps = 252):
        min_N = 3
        max_N = 6
        Ns = np.logspace(min_N, max_N, num_points, dtype=int)
        
        prices, errors = [], []
        for N in Ns:
            if option_type == "vanilla":
                fair_price, std_error, _ = self.price_vanilla_call(num_simulations=N)
            elif option_type == "knockout":
                if barrier is None or type is None:
                    raise ValueError("Need barrier and type (e.g. down-and-out)")
                fair_price, std_error, _ = self.price_knockout_call(
                    barrier=barrier, type=type,
                    num_simulations=N, n_steps=n_steps)
            else:
                raise ValueError("option_type must be vanilla or knockout")
            prices.append(fair_price)
            errors.append(std_error)

        prices, errors = np.array(prices), np.array(errors)
        return Ns, prices, errors
    
    def plot_convergence(self, option_type, num_points = 10, barrier = None, type = None, n_steps = 252):
        Ns, prices, errors = self.convergence_data(option_type=option_type, num_points=num_points,
                                                  barrier=barrier, type = type, n_steps=n_steps )
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

# Option style selection
option_style = st.radio("Option style", ["Vanilla", "Knockout"])

# Inputs common to both
S0 = st.number_input("Spot FX (S0)", value=1.10, format="%.4f")
K = st.number_input("Strike (K)", value=1.12, format="%.4f")
T = st.number_input("Expiry (T, in years)", value=1.0, format="%.2f")
rd = st.number_input("Domestic Rate (rd)", value=0.05, format="%.4f")
rf = st.number_input("Foreign Rate (rf)", value=0.03, format="%.4f")
vol = st.number_input("Volatility (vol)", value=0.10, format="%.4f")

# Knockout-specific inputs
if option_style == "Knockout":
    barrier = st.number_input("Barrier level", value=1.00, format="%.4f")
    ko_type = st.selectbox("Knockout type", ["down-and-out", "up-and-out"])

# Run model when button is pressed
if st.button("Run model"):
    model = FX_Options_Model(S0, K, T, rd, rf, vol)

    if option_style == "Vanilla":
        fair_price, std_error, ci = model.price_vanilla_call(num_simulations=100000, seed=42)

        st.write(f"Vanilla Call")
        st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
        st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

        # Convergence plot
        Ns, prices, errors = model.convergence_data(option_type="vanilla")
    else:
        fair_price, std_error, ci = model.price_knockout_call(
            barrier=barrier, type=ko_type,
            num_simulations=100000, n_steps=252, seed=42
        )

        st.write(f"Knockout Call ({ko_type}, barrier={barrier})")
        st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
        st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

        # Convergence plot
        Ns, prices, errors = model.convergence_data(
            option_type="knockout", barrier=barrier, type=ko_type, n_steps=252
        )

    fig, ax = plt.subplots()
    ax.errorbar(Ns, prices, yerr=1.96 * errors, fmt="o-", capsize=4, label="MC price ±95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Estimated call price")
    ax.set_title(f"Monte Carlo Convergence ({option_style})")
    ax.legend()
    st.pyplot(fig)
