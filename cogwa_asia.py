import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------- Logging Initialization --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(funcName)-28s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("COGWA_Asia_Vol_Model")

# ===================== 1. Black‑Scholes Pricing Formula =====================
def bs_price(S, K, T, r, q, sigma, option_type='call'):
    logger.debug(f"BS input -> S={S}, K={K}, T={T:.4f}, r={r:.4f}, q={q:.4f}, sigma={sigma:.6f}, option_type={option_type}")
    
    # Guard against invalid volatility / maturity
    if sigma <= 1e-8:
        logger.warning(f"Invalid sigma {sigma}, clamp to 1e-8")
        sigma = 1e-8
    if T <= 1e-8:
        logger.warning(f"Invalid maturity T={T}, clamp to 1e-8")
        T = 1e-8

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    c = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    if option_type.lower() == 'put':
        p = c - S * np.exp(-q*T) + K * np.exp(-r*T)
        logger.debug(f"BS result -> d1={d1:.6f}, d2={d2:.6f}, option_price={p:.4f}")
        return p

    logger.debug(f"BS result -> d1={d1:.6f}, d2={d2:.6f}, option_price={c:.4f}")
    return c

# ===================== 2. SVI / SSVI Base Function =====================
def svi_iv(k, a,b,rho,m,sigma):
    """Calculate implied volatility via SVI parametrization"""
    logger.debug(f"SVI input -> k={k:.4f}, a={a:.4f}, b={b:.4f}, rho={rho:.4f}, m={m:.4f}, sigma={sigma:.4f}")
    var = a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))
    
    if var < 1e-8:
        logger.warning(f"SVI variance negative {var}, clamp to 1e-8")
        var = 1e-8
    iv = np.sqrt(var)
    logger.debug(f"SVI output iv={iv:.6f}")
    return iv

# ===================== 3. Core Class for Asia-Pacific COGWA Model =====================
class COGWA_Asia:
    def __init__(self, tenors, deltas, iv_data, bid_ask_spread=None):
        logger.info("Initialize COGWA_Asia volatility surface model")
        self.tenors = np.array(tenors)
        self.deltas = np.array(deltas)
        self.iv_data = np.array(iv_data)
        self.bid_ask_spread = bid_ask_spread if bid_ask_spread is not None else np.zeros_like(iv_data)
        
        # Asia-Pacific enhanced regularization parameters
        self.lambda_var = 0.01          # Penalty weight for variance monotonicity
        self.lambda_reg = 0.02          # L2 regularization coefficient (Asia market exclusive)
        self.max_tail_curv = 0.3        # Upper bound for volatility tail curvature
        self.main_tenor_idx = 0         # Default bootstrap benchmark: 3-month tenor
        self.tail_delta_cutoff = 0.7    # Use SSVI extrapolation when absolute delta > 0.7
        
        # Cache to store smoothed volatility curves per tenor
        self.tenor_iv_cache = {}

        logger.info(f"Model dimension: tenors={len(self.tenors)}, deltas={len(self.deltas)}, iv_shape={self.iv_data.shape}")
        logger.info(f"Hyperparams: lambda_var={self.lambda_var}, lambda_reg={self.lambda_reg}, max_tail_curv={self.max_tail_curv}, tail_cutoff={self.tail_delta_cutoff}")

    def regularized_cubic_smooth(self, x, y, spread):
        """
        Cubic smoothing with L2 regularization & bid-ask spread weighting
        Optimized for sparse, noisy market option quotes
        """
        logger.debug(f"Start cubic smooth, x count={len(x)}, y count={len(y)}")
        # Weight inversely proportional to bid-ask spread; wider spread = lower weight
        weight = 1.0 / (1.0 + spread)
        f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        x_fine = np.linspace(-1, 1, 200)
        iv_raw = f(x_fine)
        
        # L2 regularization loss: penalize excessive second derivative oscillations
        def loss(smooth_iv):
            deriv2 = np.gradient(np.gradient(smooth_iv))
            reg_loss = self.lambda_reg * np.sum(deriv2 ** 2)
            fit_loss = np.sum(weight * (np.interp(x, x_fine, smooth_iv) - y) ** 2)
            total = fit_loss + reg_loss
            return total
        
        logger.info("Launch L-BFGS-B optimization for cubic smoothing curve")
        res = minimize(loss, iv_raw, method='L-BFGS-B')
        iv_smooth = res.x
        logger.info(f"Smooth optimization finished, success={res.success}, final_loss={res.fun:.6f}")
        
        # Enforce maximum curvature constraint on left & right volatility tails
        clip_cnt = 0
        for i in range(10):
            if iv_smooth[-i-1] - iv_smooth[-i-2] > self.max_tail_curv:
                iv_smooth[-i-1] = iv_smooth[-i-2] + self.max_tail_curv
                clip_cnt += 1
            if iv_smooth[i+1] - iv_smooth[i] > self.max_tail_curv:
                iv_smooth[i+1] = iv_smooth[i] + self.max_tail_curv
                clip_cnt += 1
        if clip_cnt > 0:
            logger.warning(f"Tail curvature clipped {clip_cnt} times to max limit {self.max_tail_curv}")
        
        logger.debug("Cubic smooth complete, return fine grid & smoothed IV array")
        return x_fine, iv_smooth

    def enforce_variance_monotonic_bootstrap(self):
        """
        Bootstrap calibration: enforce T*σ² total variance increases monotonically with maturity
        Constraints propagate from the benchmark tenor to all other expirations
        """
        logger.info("Start variance monotonic bootstrap calibration")
        main_T = self.tenors[self.main_tenor_idx]
        main_iv = self.iv_data[self.main_tenor_idx]
        main_var = main_T * np.array(main_iv) ** 2
        logger.debug(f"Benchmark tenor T={main_T}, main variance array shape={main_var.shape}")

        adjust_count = 0
        for idx, T in enumerate(self.tenors):
            if idx == self.main_tenor_idx:
                logger.debug(f"Skip benchmark tenor idx={idx}")
                continue
            iv = self.iv_data[idx]
            var = T * np.array(iv) ** 2
            for i in range(len(var)):
                # Longer maturity cannot have lower total variance than benchmark
                if T > main_T and var[i] < main_var[i]:
                    iv[i] = np.sqrt(main_var[i] / T)
                    adjust_count += 1
                # Shorter maturity cannot have higher total variance than benchmark
                elif T < main_T and var[i] > main_var[i]:
                    iv[i] = np.sqrt(main_var[i] / T)
                    adjust_count += 1
            self.iv_data[idx] = iv
        if adjust_count > 0:
            logger.warning(f"Variance monotonic adjustment modified {adjust_count} IV points")
        else:
            logger.info("No IV adjustment needed for variance monotonicity")
        return self.iv_data

    def fit_single_tenor_curve(self, tenor_idx):
        logger.debug(f"Request smooth curve for tenor index={tenor_idx}")
        # Return cached smooth curve if already computed
        if tenor_idx in self.tenor_iv_cache:
            logger.info(f"Hit cache for tenor {tenor_idx}, skip recalculation")
            return self.tenor_iv_cache[tenor_idx]
        
        logger.info(f"Cache miss, compute cubic smooth curve for tenor {tenor_idx}")
        x_fine, iv_smooth = self.regularized_cubic_smooth(
            self.deltas,
            self.iv_data[tenor_idx],
            self.bid_ask_spread[tenor_idx]
        )
        self.tenor_iv_cache[tenor_idx] = (x_fine, iv_smooth)
        logger.info(f"Store smooth curve to cache, tenor_idx={tenor_idx}")
        return x_fine, iv_smooth

    def get_iv(self, T, delta, S0, K, svi_params=None):
        """
        Piecewise hybrid volatility interpolation logic:
        - |delta| < 0.7: Regularized COGWA cubic interpolation
        - |delta| >= 0.7: SSVI tail extrapolation for deep OTM strikes
        """
        logger.info(f"Get hybrid IV, target T={T:.4f}, target delta={delta:.4f}, abs_delta={abs(delta):.4f}")
        self.enforce_variance_monotonic_bootstrap()
        
        # Select tenor with maturity closest to target T
        idx = np.argmin(np.abs(self.tenors - T))
        logger.debug(f"Matched nearest tenor index={idx}, tenor value={self.tenors[idx]}")
        x_fine, iv_smooth = self.fit_single_tenor_curve(idx)
        
        # At-the-money & moderate delta region: use smoothed COGWA interpolation
        if abs(delta) < self.tail_delta_cutoff:
            vol = np.interp(delta, x_fine, iv_smooth)
            logger.info(f"Mid delta region, use cubic interpolated vol={vol:.6f}")
            return vol
        
        # Deep out-of-the-money tail region: activate SSVI extrapolation
        # For Asia market with no liquid quotes on far OTM options
        if svi_params is None:
            vol = np.interp(delta, x_fine, iv_smooth)
            logger.warning(f"No SVI params provided for tail, fallback to cubic interp vol={vol:.6f}")
            return vol
        
        a,b,rho,m,sigma = svi_params
        logger.debug(f"Enter SSVI tail extrapolation, SVI params loaded")
        # Correct log moneyness k=ln(K/S0)
        k = np.log(K / S0)
        vol_tail = svi_iv(k, a,b,rho,m,sigma)
        logger.info(f"Deep OTM tail, SSVI extrapolated vol={vol_tail:.6f}")
        return vol_tail

def build_default_model():
    logger.info("Build default COGWA Asia pricing model")
    tenors = [0.25, 1.0]
    deltas = [-0.9, -0.25, 0, 0.25, 0.9]
    raw_iv = np.array([
        [0.26, 0.21, 0.19, 0.20, 0.285],
        [0.24, 0.20, 0.18, 0.19, 0.25]
    ])
    bid_ask_spread = np.array([
        [0.08, 0.03, 0.02, 0.03, 0.10],
        [0.07, 0.02, 0.02, 0.03, 0.09]
    ])
    svi_params = (0.012, 0.12, -0.65, 0.0, 0.22)
    model = COGWA_Asia(tenors, deltas, raw_iv, bid_ask_spread)
    return model, svi_params


def price_hybrid_option(model, T, delta, S0, K, q, r, option_type, svi_params):
    vol = model.get_iv(T, delta, S0, K, svi_params)
    price = bs_price(S0, K, T, r, q, vol, option_type=option_type)
    return vol, price


def price_pure_svi(S0, K, T, q, r, option_type, svi_params):
    k = np.log(K / S0)
    vol = svi_iv(k, *svi_params)
    price = bs_price(S0, K, T, r, q, vol, option_type=option_type)
    return vol, price


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class COGWAAsiaPricingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("COGWA Asia Option Pricing")
        self.model, self.svi_params = build_default_model()
        self.history = []

        self._build_widgets()
        self._set_default_inputs()
        self._update_chart()

    def _build_widgets(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0
        for label_text, var_name in [
            ("Spot price (S0)", "spot"),
            ("Strike price (K)", "strike"),
            ("Time to maturity (years)", "maturity"),
            ("Delta", "delta"),
            ("Dividend yield (q)", "div"),
            ("Desk discount rate (r desk)", "r_desk"),
            ("Public discount rate (r public)", "r_public")
        ]:
            ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(frame, width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            setattr(self, f"entry_{var_name}", entry)
            row += 1

        ttk.Label(frame, text="Option type").grid(row=row, column=0, sticky="w", pady=4)
        self.option_type = ttk.Combobox(frame, values=["call", "put"], state="readonly", width=16)
        self.option_type.grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Button(frame, text="Price Option", command=self.on_price).grid(row=row, column=0, pady=12)
        ttk.Button(frame, text="Save History", command=self.save_history_csv).grid(row=row, column=1, pady=12)
        row += 1

        separator = ttk.Separator(frame, orient="horizontal")
        separator.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        self.result_text = tk.Text(frame, width=48, height=8, wrap="word", state="disabled")
        self.result_text.grid(row=row, column=0, columnspan=2, sticky="nsew")
        row += 1

        chart_frame = ttk.Frame(frame)
        chart_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=8)
        frame.rowconfigure(row, weight=1)

        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Pricing History")
        self.ax.set_xlabel("Run")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_default_inputs(self):
        self.entry_spot.insert(0, "5000")
        self.entry_strike.insert(0, "4500")
        self.entry_maturity.insert(0, "1.0")
        self.entry_delta.insert(0, "-0.9")
        self.entry_div.insert(0, "0.015")
        self.entry_r_desk.insert(0, "0.0485")
        self.entry_r_public.insert(0, "0.045")
        self.option_type.set("put")

    def on_price(self):
        inputs = {
            "spot": parse_float(self.entry_spot.get()),
            "strike": parse_float(self.entry_strike.get()),
            "maturity": parse_float(self.entry_maturity.get()),
            "delta": parse_float(self.entry_delta.get()),
            "div": parse_float(self.entry_div.get()),
            "r_desk": parse_float(self.entry_r_desk.get()),
            "r_public": parse_float(self.entry_r_public.get())
        }

        if any(value is None for value in inputs.values()):
            messagebox.showerror("Input error", "Please enter valid numeric values in all fields.")
            return

        option_type = self.option_type.get().lower()
        if option_type not in ["call", "put"]:
            messagebox.showerror("Input error", "Option type must be 'call' or 'put'.")
            return

        hybrid_vol, hybrid_price = price_hybrid_option(
            self.model,
            inputs["maturity"],
            inputs["delta"],
            inputs["spot"],
            inputs["strike"],
            inputs["div"],
            inputs["r_desk"],
            option_type,
            self.svi_params
        )

        svi_vol, svi_price = price_pure_svi(
            inputs["spot"],
            inputs["strike"],
            inputs["maturity"],
            inputs["div"],
            inputs["r_public"],
            option_type,
            self.svi_params
        )

        output = (
            f"COGWA-Asia Hybrid Volatility: {hybrid_vol:.4f}\n"
            f"COGWA-Asia Hybrid Price:      {hybrid_price:.2f}\n"
            f"Pure SVI Volatility:          {svi_vol:.4f}\n"
            f"Pure SVI Price:               {svi_price:.2f}\n"
            f"Nearest tenor used:           {self.model.tenors[np.argmin(np.abs(self.model.tenors - inputs['maturity']))]:.2f} years\n"
            f"SVI tail parameters:          {self.svi_params}\n"
        )

        self.history.append({
            "run": len(self.history) + 1,
            "spot": inputs["spot"],
            "strike": inputs["strike"],
            "maturity": inputs["maturity"],
            "delta": inputs["delta"],
            "option_type": option_type,
            "hybrid_vol": hybrid_vol,
            "hybrid_price": hybrid_price,
            "svi_vol": svi_vol,
            "svi_price": svi_price
        })
        self._display_result(output)
        self._update_chart()

    def _display_result(self, text):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def _update_chart(self):
        self.ax.clear()
        self.ax.set_title("Pricing History")
        self.ax.set_xlabel("Run")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)

        if self.history:
            runs = [entry["run"] for entry in self.history]
            hybrid_prices = [entry["hybrid_price"] for entry in self.history]
            svi_prices = [entry["svi_price"] for entry in self.history]
            self.ax.plot(runs, hybrid_prices, marker="o", label="Hybrid Price")
            self.ax.plot(runs, svi_prices, marker="s", label="Pure SVI Price")
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, "No pricing history yet.", ha="center", va="center", transform=self.ax.transAxes)

        self.canvas.draw()

    def save_history_csv(self):
        if not self.history:
            messagebox.showinfo("Save history", "No pricing history to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save pricing history"
        )
        if not file_path:
            return

        import csv
        with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "run", "spot", "strike", "maturity", "delta", "option_type",
                "hybrid_vol", "hybrid_price", "svi_vol", "svi_price"
            ])
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

        messagebox.showinfo("Save history", f"Pricing history saved to {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = COGWAAsiaPricingUI(root)
    root.mainloop()
