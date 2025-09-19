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

    def _mc_simulation(self, num_simulations, n_steps=1, reduce_variance=True, seed=None):
        rng = np.random.default_rng(seed)
        n = num_simulations // (2 if reduce_variance else 1)  # half if we take Z and -Z
        Z = rng.standard_normal((n, n_steps))
        if reduce_variance:
            Z = np.vstack([Z, -Z])  # concatenate antithetics
        return Z

    def price_vanilla(self, call_or_put="call", num_simulations=100000, reduce_variance=True, seed=None):
        Z = self._mc_simulation(num_simulations=num_simulations, n_steps=1,
                                reduce_variance=reduce_variance, seed=seed).flatten()

        drift = (self.rd - self.rf - 0.5 * self.vol**2) * self.T
        sigma_Z = self.vol * np.sqrt(self.T) * Z
        S_t = self.S0 * np.exp(drift + sigma_Z)

        if call_or_put == "call":
            payoff = np.maximum(S_t - self.K, 0.0)
        elif call_or_put == "put":
            payoff = np.maximum(self.K - S_t, 0.0)
        else:
            raise ValueError("option must be call or put")

        discounted = np.exp(-self.rd * self.T) * payoff
        fair_price = np.mean(discounted)
        std_error = discounted.std(ddof=1) / np.sqrt(Z.size)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)
        # 1.96 = critical value of 95% confidence interval
        return fair_price, std_error, conf_interval

    def price_knockout(self, barrier, ko_type, call_or_put="call",
                       num_simulations=10000, n_steps=252, reduce_variance=True, seed=None):
    
        # need ko_type: 'down-and-out', 'up-and-out', 'down-and-in', or 'up-and-in', and a barrier
        if barrier <= 0:
            raise ValueError("Barrier must be > 0 for log-price Brownian bridge.")

        dt = self.T / n_steps
        mu_dt = (self.rd - self.rf - 0.5 * self.vol**2) * dt
        sig_sqrt_dt = self.vol * np.sqrt(dt)
        sig2dt = (self.vol**2) * dt

        # Pre-generate all shocks with antithetics (uses reduce_variance)
        Z = self._mc_simulation(num_simulations=num_simulations, n_steps=n_steps,
                                reduce_variance=reduce_variance, seed=seed)
        n_paths = Z.shape[0]
        # Separate RNG for uniforms to keep determinism
        rng_u = np.random.default_rng(None if seed is None else seed + 1)

        S = np.full(n_paths, self.S0, dtype=float)
        ever_hit = np.zeros(n_paths, dtype=bool)  # has the barrier been hit at least once
        logB = np.log(barrier)
        # Initial t=0 check
        if "down" in ko_type:
            ever_hit |= (S <= barrier)
        elif "up" in ko_type:
            ever_hit |= (S >= barrier)
        else:
            raise ValueError("ko_type must be 'down-and-out', 'up-and-out', 'down-and-in', or 'up-and-in'")

        for k in range(n_steps):
            S_old = S.copy()
            Zk = Z[:, k]
            S = S * np.exp(mu_dt + sig_sqrt_dt * Zk)

            # Gridpoint crossing (inclusive): <= for down, >= for up
            if "down" in ko_type:
                ever_hit |= (S <= barrier)
            elif "up" in ko_type:
                ever_hit |= (S >= barrier)
            else:
                raise ValueError("ko_type must be 'down-and-out', 'up-and-out', 'down-and-in', or 'up-and-in'")

            # Brownian bridge correction on paths not marked as hit
            idx = np.where(~ever_hit)[0]
            if idx.size:
                x0 = np.log(S_old[idx])
                x1 = np.log(S[idx])

                if sig2dt == 0.0:
                    p = np.zeros_like(x0)
                elif "down" in ko_type:
                    # Hit down-barrier between x0 and x1
                    m = np.minimum(x0, x1)
                    p = np.exp(-2.0 * (x0 - logB) * (x1 - logB) / sig2dt)
                    p = np.where(logB <= m, p, 0.0)
                else:  # "up"
                    # Hit up-barrier between x0 and x1
                    m = np.maximum(x0, x1)
                    p = np.exp(-2.0 * (logB - x0) * (logB - x1) / sig2dt)
                    p = np.where(logB >= m, p, 0.0)

                u = rng_u.random(idx.size)
                hit_inside = (u < p)
                if np.any(hit_inside):
                    ever_hit[idx[hit_inside]] = True

        # Payoff at maturity
        if call_or_put == "call":
            payoff_base = np.maximum(S - self.K, 0.0)
        elif call_or_put == "put":
            payoff_base = np.maximum(self.K - S, 0.0)
        else:
            raise ValueError("option must be 'call' or 'put'")

        # In/Out:
        if "out" in ko_type:
            payoff = np.where(~ever_hit, payoff_base, 0.0)  # out: worthless if ever hit
        else:  # i.e. "in"
            payoff = np.where(ever_hit, payoff_base, 0.0)   # in : worthless unless hit

        discounted = np.exp(-self.rd * self.T) * payoff
        fair_price = np.mean(discounted)
        std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)
        return fair_price, std_error, conf_interval

    def price_vol_knockout(self, vol_barrier, call_or_put="call",
                           num_simulations=100000, n_steps=252, reduce_variance=True, seed=None):
        # Standard vol KO: knocks out when annualised realised vol >= vol_barrier
        dt = self.T / n_steps

        Z = self._mc_simulation(num_simulations, n_steps, reduce_variance, seed)
        n_paths = Z.shape[0]

        mu_dt = (self.rd - self.rf - 0.5 * self.vol**2) * dt
        sig_sqrt_dt = self.vol * np.sqrt(dt)

        S = np.full(n_paths, self.S0, float)
        alive = np.ones(n_paths, dtype=bool)  # not knocked out yet by vol
        cum_vol = np.zeros(n_paths, float)

        budget_step = (vol_barrier ** 2) * dt
        cum_budget = 0.0

        for k in range(n_steps):
            idx = np.where(alive)[0]
            if idx.size == 0:
                break

            S_old = S[idx]
            Zk = Z[idx, k]
            S_new = S_old * np.exp(mu_dt + sig_sqrt_dt * Zk)

            dlog = np.log(S_new) - np.log(S_old)
            cum_vol[idx] += dlog * dlog
            S[idx] = S_new

            cum_budget += budget_step
            ko_now = (cum_vol[idx] >= cum_budget)
            if np.any(ko_now):
                alive[idx[ko_now]] = False

        if call_or_put == "call":
            payoff_base = np.maximum(S - self.K, 0.0)
        elif call_or_put == "put":
            payoff_base = np.maximum(self.K - S, 0.0)
        else:
            raise ValueError("option must be 'call' or 'put'")

        payoff = np.where(alive, payoff_base, 0.0)
        discounted = np.exp(-self.rd * self.T) * payoff
        fair_price = discounted.mean()
        std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
        conf_interval = (fair_price - 1.96 * std_error, fair_price + 1.96 * std_error)
        return fair_price, std_error, conf_interval

    def convergence_data(self, option_type, num_points=8, barrier=None,
                         ko_type=None, n_steps=252, vol_barrier=None, call_or_put="call"):
        if option_type in ("knockout", "VKO"):
            min_N, max_N = 3, 5   # 1e3 … 1e5 to avoid OOM
        else:
            min_N, max_N = 3, 6   # 1e3 … 1e6 for vanilla
            Ns = np.logspace(min_N, max_N, num_points, dtype=int)

        prices, errors = [], []
        for N in Ns:
            if option_type == "vanilla":
                fair_price, std_error, _ = self.price_vanilla(call_or_put=call_or_put, num_simulations=N)
            elif option_type == "knockout":
                if barrier is None or ko_type is None:
                    raise ValueError("Need barrier and type (e.g. down-and-out)")
                fair_price, std_error, _ = self.price_knockout(
                    barrier=barrier, ko_type=ko_type, call_or_put=call_or_put,
                    num_simulations=N, n_steps=n_steps
                )
            elif option_type == "VKO":
                if vol_barrier is None:
                    raise ValueError("Need realised vol barrier for VKO")
                fair_price, std_error, _ = self.price_vol_knockout(
                    vol_barrier=vol_barrier, call_or_put=call_or_put,
                    num_simulations=N, n_steps=n_steps
                )
            else:
                raise ValueError("option_type must be vanilla or knockout")

            prices.append(fair_price)
            errors.append(std_error)

        prices, errors = np.array(prices), np.array(errors)
        return Ns, prices, errors

    def plot_convergence(self, option_type, num_points=10, barrier=None, ko_type=None,
                         n_steps=252, vol_barrier=None, call_or_put="call"):
        Ns, prices, errors = self.convergence_data(
            option_type=option_type, num_points=num_points,
            barrier=barrier, ko_type=ko_type, n_steps=n_steps,
            vol_barrier=vol_barrier, call_or_put=call_or_put
        )
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
option_style = st.radio("Option style", ["Vanilla", "Knockout", "Volatility Knockout"])

# Call/Put selection
option_side = st.radio("Option type", ["call", "put"])

# Common inputs
S0 = st.number_input("Spot FX (S0)", value=1.10, format="%.4f")
K = st.number_input("Strike (K)", value=1.12, format="%.4f")
T = st.number_input("Expiry (T, in years)", value=1.0, format="%.2f")
rd = st.number_input("Domestic Rate (rd)", value=0.05, format="%.4f")
rf = st.number_input("Foreign Rate (rf)", value=0.03, format="%.4f")
vol = st.number_input("Volatility (vol)", value=0.10, format="%.4f")

# Knockout-specific inputs
if option_style == "Knockout":
    barrier = st.number_input("Barrier level", value=1.00, format="%.4f")
    ko_type = st.selectbox("Knockout type", ["down-and-out", "up-and-out", "down-and-in", "up-and-in"])

# Volatility KO-specific inputs
if option_style == "Volatility Knockout":
    H = st.number_input("Volatility KO threshold (annualised, e.g. 0.30 = 30%)",
                        value=0.30, format="%.4f")

# Run model
if st.button("Run model"):
    model = FX_Options_Model(S0, K, T, rd, rf, vol)

    if option_style == "Vanilla":
        fair_price, std_error, ci = model.price_vanilla(
            call_or_put=option_side, num_simulations=100000, seed=42
        )
        st.write(f"Vanilla {option_side.capitalize()}")
        st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
        st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

        Ns, prices, errors = model.convergence_data(
            option_type="vanilla", call_or_put=option_side
        )

    elif option_style == "Knockout":
        fair_price, std_error, ci = model.price_knockout(
            barrier=barrier, ko_type=ko_type, call_or_put=option_side,
            num_simulations=100000, n_steps=252, seed=42
        )
        st.write(f"{ko_type} {option_side.capitalize()} (barrier={barrier})")
        st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
        st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

        Ns, prices, errors = model.convergence_data(
            option_type="knockout", barrier=barrier, ko_type=ko_type,
            n_steps=252, call_or_put=option_side
        )

    elif option_style == "Volatility Knockout":
        fair_price, std_error, ci = model.price_vol_knockout(
            vol_barrier=H, call_or_put=option_side,
            num_simulations=100000, n_steps=252, seed=42
        )
        st.write(f"Volatility Knockout {option_side.capitalize()} (H={H:.2%})")
        st.write(f"Fair price: {fair_price:.6f}, Standard error: {std_error:.6f}")
        st.write(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")

        Ns, prices, errors = model.convergence_data(
            option_type="VKO", vol_barrier=H, n_steps=252, call_or_put=option_side
        )

    # Shared convergence plot
    fig, ax = plt.subplots()
    ax.errorbar(Ns, prices, yerr=1.96 * errors, fmt="o-", capsize=4, label="MC price ±95% CI")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Estimated option price")
    ax.set_title(f"Monte Carlo Convergence ({option_style}, {option_side})")
    ax.legend()
    st.pyplot(fig)
