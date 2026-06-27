import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ===================== 1. Black‑Scholes Pricing Formula =====================
def bs_price(S, K, T, r, q, sigma):
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    c = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return c

# ===================== 2. SVI / SSVI Base Function =====================
def svi_iv(k, a,b,rho,m,sigma):
    """Calculate implied volatility via SVI parametrization"""
    return np.sqrt(a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2)))

# ===================== 3. Core Class for Asia-Pacific COGWA Model =====================
class COGWA_Asia:
    def __init__(self, tenors, deltas, iv_data, bid_ask_spread=None):
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

    def regularized_cubic_smooth(self, x, y, spread):
        """
        Cubic smoothing with L2 regularization & bid-ask spread weighting
        Optimized for sparse, noisy market option quotes
        """
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
            return fit_loss + reg_loss
        
        res = minimize(loss, iv_raw, method='L-BFGS-B')
        iv_smooth = res.x
        
        # Enforce maximum curvature constraint on left & right volatility tails
        for i in range(10):
            if iv_smooth[-i-1] - iv_smooth[-i-2] > self.max_tail_curv:
                iv_smooth[-i-1] = iv_smooth[-i-2] + self.max_tail_curv
            if iv_smooth[i+1] - iv_smooth[i] > self.max_tail_curv:
                iv_smooth[i+1] = iv_smooth[i] + self.max_tail_curv
        
        return x_fine, iv_smooth

    def enforce_variance_monotonic_bootstrap(self):
        """
        Bootstrap calibration: enforce T*σ² total variance increases monotonically with maturity
        Constraints propagate from the benchmark tenor to all other expirations
        """
        main_T = self.tenors[self.main_tenor_idx]
        main_iv = self.iv_data[self.main_tenor_idx]
        main_var = main_T * np.array(main_iv) ** 2
        
        for idx, T in enumerate(self.tenors):
            if idx == self.main_tenor_idx:
                continue
            iv = self.iv_data[idx]
            var = T * np.array(iv) ** 2
            for i in range(len(var)):
                # Longer maturity cannot have lower total variance than benchmark
                if T > main_T and var[i] < main_var[i]:
                    iv[i] = np.sqrt(main_var[i] / T)
                # Shorter maturity cannot have higher total variance than benchmark
                elif T < main_T and var[i] > main_var[i]:
                    iv[i] = np.sqrt(main_var[i] / T)
            self.iv_data[idx] = iv
        return self.iv_data

    def fit_single_tenor_curve(self, tenor_idx):
        # Return cached smooth curve if already computed
        if tenor_idx in self.tenor_iv_cache:
            return self.tenor_iv_cache[tenor_idx]
        x_fine, iv_smooth = self.regularized_cubic_smooth(
            self.deltas,
            self.iv_data[tenor_idx],
            self.bid_ask_spread[tenor_idx]
        )
        self.tenor_iv_cache[tenor_idx] = (x_fine, iv_smooth)
        return x_fine, iv_smooth

    def get_iv(self, T, delta, S0, K, svi_params=None):
        """
        Piecewise hybrid volatility interpolation logic:
        - |delta| < 0.7: Regularized COGWA cubic interpolation
        - |delta| >= 0.7: SSVI tail extrapolation for deep OTM strikes
        """
        self.enforce_variance_monotonic_bootstrap()
        
        # Select tenor with maturity closest to target T
        idx = np.argmin(np.abs(self.tenors - T))
        x_fine, iv_smooth = self.fit_single_tenor_curve(idx)
        
        # At-the-money & moderate delta region: use smoothed COGWA interpolation
        if abs(delta) < self.tail_delta_cutoff:
            return np.interp(delta, x_fine, iv_smooth)
        
        # Deep out-of-the-money tail region: activate SSVI extrapolation
        # For Asia market with no liquid quotes on far OTM options
        if svi_params is None:
            return np.interp(delta, x_fine, iv_smooth)
        a,b,rho,m,sigma = svi_params
        # 修复：SVI模型正确的对数执行价计算 k=ln(K/S0)，无负数对数问题
        k = np.log(K / S0)
        vol_tail = svi_iv(k, a,b,rho,m,sigma)
        return vol_tail

# ===================== 4. Calibration Parameters for Asia Market (HSI/A50 Style Benchmark) =====================
S0 = 5000
q = 0.015
r_public = 0.045
r_ms = 0.0485

# Maturity tenors: 3-month / 1-year (typical liquidity gap structure in Asia equity derivatives)
tenors = [0.25, 1.0]
deltas = [-0.9, -0.25, 0, 0.25, 0.9]

# Raw noisy implied volatility surface + wide bid-ask spreads simulating Asia market liquidity
raw_iv = np.array([
    [0.26, 0.21, 0.19, 0.20, 0.285],
    [0.24, 0.20, 0.18, 0.19, 0.25]
])
bid_ask_spread = np.array([
    [0.08, 0.03, 0.02, 0.03, 0.10],
    [0.07, 0.02, 0.02, 0.03, 0.09]
])

# SVI constant parameters for deep OTM tail extrapolation
svi_params = (0.012, 0.12, -0.65, 0.0, 0.22)

# Initialize Asia-Pacific COGWA volatility model instance
cogwa_asia = COGWA_Asia(tenors, deltas, raw_iv, bid_ask_spread)

# Pricing target: 1-year 90Δ deep OTM put option
T_target = 1.0
delta_target = -0.9
K_target = 4500

# Fetch hybrid COGWA tail volatility and compute option price（传入S0和K_target修复参数）
cogwa_vol = cogwa_asia.get_iv(T_target, delta_target, S0, K_target, svi_params)
price_cogwa = bs_price(S0, K_target, T_target, r_ms, q, cogwa_vol)

# Benchmark: Pure SVI volatility pricing for comparison
k = np.log(K_target/S0)
svi_vol = svi_iv(k, *svi_params)
price_svi = bs_price(S0, K_target, T_target, r_public, q, svi_vol)

# Print output comparison results
print(f"COGWA‑Asia Hybrid Tail Volatility: {cogwa_vol:.4f}")
print(f"Pure SVI Fitted Volatility:        {svi_vol:.4f}")
print(f"COGWA‑Asia Prop Desk Price:         {price_cogwa:.2f}")
print(f"SVI Price (Public Discount Rate):   {price_svi:.2f}")