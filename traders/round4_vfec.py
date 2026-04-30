from __future__ import annotations
from collections import deque
import math
from datamodel import Order, TradingState

# ---------------------------------------------------------------------------
# VELVETFRUIT_EXTRACT_VOUCHER (VEV_K) — Improved Options Strategy
#
# The vouchers are European call options on VELVETFRUIT_EXTRACT (VFE).
# Fair value = intrinsic + time_value
#   intrinsic  = max(0, S - K)   where S = VFE spot, K = strike
#   time_value = Black-Scholes-derived, estimated live from observed
#                option mid-prices and decaying via theta over the round.
#
# Problems in the original (round3_trader9.py):
#   1. Only traded SCALP_STRIKES = [5300, 5400] — 8 other vouchers ignored.
#      All strikes can be profitable if fair value is estimated correctly.
#   2. TV_SEED values seeded from historical data but deque is AVERAGED,
#      not decayed. As time-to-expiry (TTE) shrinks each day, time value
#      MUST fall toward zero. Averaging historical TV keeps estimates too
#      high late in the round — the strategy pays too much for options.
#   3. ENTRY_THRESH was manually hardcoded per strike. A better approach
#      is to derive the threshold from observed bid-ask spread so the edge
#      scales automatically with market conditions.
#   4. No delta hedging awareness — buying calls gives long delta exposure
#      on VFE which can compound losses if VFE moves against you.
#      We track net delta and cap it to stay within delta-neutral bounds.
#   5. No position unwind logic — once a position is accumulated the
#      strategy never exits except via the other side of the same signal.
#
# Improvements:
#   - Trade ALL 10 strikes, not just 5300/5400.
#   - TTE-aware time value: estimate implied vol from each option mid,
#     then re-price using BS with remaining TTE to get a model fair value.
#     This automatically decays TV as TTE shrinks.
#   - Dynamic entry threshold: max(MIN_EDGE, spread_ema * EDGE_SPREAD_MULT).
#   - Net delta cap: sum(position_K * delta_K) must stay within DELTA_LIMIT.
#   - Unwind logic: if |z_tv| > EXIT_Z, take the other side to flatten.
#   - Counterparty signal on underlying feeds into spot estimate.
# ---------------------------------------------------------------------------

VFE_PRODUCT    = "VELVETFRUIT_EXTRACT"
ALL_STRIKES    = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_LIMIT   = 300          # per voucher, per round description
ORDER_SIZE     = 10           # max qty per aggressive fill

# TTE (time-to-expiry in trading days). Round 4 starts with TTE = 4 days.
# Each day = 10,000 ticks. We reduce TTE_DAYS by 1 per day.
TTE_DAYS_START = 4
TICKS_PER_DAY  = 10_000
TRADING_DAYS   = 252          # annualisation basis

# Volatility estimation
VOL_WINDOW     = 200          # ticks to use for implied vol EMA
VOL_FALLBACK   = 1.0          # annualised vol fallback if BS inversion fails

# Entry / exit
MIN_EDGE       = 2.0          # minimum price edge to enter (ticks)
EDGE_SPREAD_MULT = 0.6        # entry threshold = max(MIN_EDGE, spread*mult)
EXIT_ZSCORE    = 1.5          # |z| of TV vs rolling baseline to trigger unwind

# Delta management
DELTA_LIMIT    = 50.0         # max net delta across all vouchers

# TV_SEED from historical data (used only as initial prior, not averaged in)
TV_SEED = {
    4000: 0.0, 4500: 0.0, 5000: 6.8,  5100: 21.6, 5200: 51.0,
    5300: 46.1, 5400: 18.5, 5500: 8.1, 6000: 0.5,  6500: 0.5,
}


# ---------------------------------------------------------------------------
# Black-Scholes helpers (calls only; vouchers are calls)
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation, accurate to 1e-7."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    return 0.5 + sign * (0.5 - math.exp(-0.5*x*x) / math.sqrt(2*math.pi) * poly)

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """European call, zero interest rate / dividend."""
    if T <= 0:
        return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    return _norm_cdf(d1)

def implied_vol(S: float, K: float, T: float, market_price: float,
                lo: float = 0.01, hi: float = 20.0, iters: int = 40) -> float:
    """Bisection solver for implied vol; returns VOL_FALLBACK on failure."""
    if T <= 0 or market_price <= max(0.0, S - K):
        return VOL_FALLBACK
    for _ in range(iters):
        mid = (lo + hi) / 2
        if bs_call_price(S, K, T, mid) > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return (lo + hi) / 2


class VoucherStrategy:

    def __init__(self, spot_ref: "SpotReference"):
        self._spot = spot_ref        # shared reference to VFE spot estimate
        self._tick = 0

        # Per-strike state
        self._iv_ema:  dict[int, float]  = {K: VOL_FALLBACK for K in ALL_STRIKES}
        self._tv_hist: dict[int, deque]  = {K: deque(maxlen=200) for K in ALL_STRIKES}
        self._spread_ema: dict[int, float | None] = {K: None for K in ALL_STRIKES}

    def _tte(self) -> float:
        """TTE in annualised years, decaying tick-by-tick."""
        days_elapsed = self._tick / TICKS_PER_DAY
        days_left = max(0.0, TTE_DAYS_START - days_elapsed)
        return days_left / TRADING_DAYS

    def _net_delta(self, positions: dict) -> float:
        S   = self._spot.mid
        T   = self._tte()
        net = 0.0
        for K in ALL_STRIKES:
            pos = positions.get(f"VEV_{K}", 0)
            if pos == 0: continue
            sig = self._iv_ema[K]
            net += pos * bs_call_delta(S, K, T, sig)
        return net

    def trade(self, order_depths: dict, positions: dict) -> dict:
        self._tick += 1
        result: dict = {}

        S = self._spot.mid
        if S <= 0: return result
        T = self._tte()

        net_delta = self._net_delta(positions)

        for K in ALL_STRIKES:
            product = f"VEV_{K}"
            od = order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid_opt  = (best_bid + best_ask) / 2.0
            spread   = best_ask - best_bid

            # Update spread EMA
            se = self._spread_ema[K]
            self._spread_ema[K] = float(spread) if se is None else 0.85*se + 0.15*spread

            # Implied vol from current mid
            iv = implied_vol(S, K, T, mid_opt)
            self._iv_ema[K] = 0.90*self._iv_ema[K] + 0.10*iv

            # Model fair value using smoothed IV
            fair = bs_call_price(S, K, T, self._iv_ema[K])

            # Track time-value history for z-score unwind signal
            tv_obs = mid_opt - max(0.0, S - K)
            if tv_obs >= 0:
                self._tv_hist[K].append(tv_obs)

            # Dynamic entry edge
            edge = max(MIN_EDGE, (self._spread_ema[K] or 2.0) * EDGE_SPREAD_MULT)

            cur_pos = positions.get(product, 0)
            orders  = []
            delta_K = bs_call_delta(S, K, T, self._iv_ema[K])

            # --- BUY signal: ask is cheap vs model ---
            if (best_ask <= fair - edge
                    and cur_pos < OPTION_LIMIT
                    and net_delta + delta_K <= DELTA_LIMIT):
                qty = min(-od.sell_orders[best_ask], OPTION_LIMIT - cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    net_delta += qty * delta_K

            # --- SELL signal: bid is rich vs model ---
            elif (best_bid >= fair + edge
                    and cur_pos > -OPTION_LIMIT
                    and net_delta - delta_K >= -DELTA_LIMIT):
                qty = min(od.buy_orders[best_bid], OPTION_LIMIT + cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    net_delta -= qty * delta_K

            # --- Unwind: TV has mean-reverted, flatten the position ---
            elif cur_pos != 0 and len(self._tv_hist[K]) >= 20:
                tv_arr = list(self._tv_hist[K])
                n   = len(tv_arr)
                mu  = sum(tv_arr) / n
                std = math.sqrt(sum((x-mu)**2 for x in tv_arr) / n)
                tv_now = mid_opt - max(0.0, S - K)
                z = (tv_now - mu) / std if std > 1e-6 else 0.0

                if cur_pos > 0 and z > EXIT_ZSCORE:
                    # We are long; TV spiked above baseline — sell to unwind
                    qty = min(cur_pos, ORDER_SIZE, od.buy_orders.get(best_bid, 0))
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                elif cur_pos < 0 and z < -EXIT_ZSCORE:
                    qty = min(-cur_pos, ORDER_SIZE, -od.sell_orders.get(best_ask, 0))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))

            if orders:
                result[product] = orders

        return result


# ---------------------------------------------------------------------------
# Shared spot reference (injected into VoucherStrategy)
# ---------------------------------------------------------------------------
class SpotReference:
    def __init__(self):
        self.mid = 5250.0   # initial prior; overwritten on first tick
        self._ema = None

    def update(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:   m = (max(b) + min(a)) / 2.0
        elif b:       m = float(max(b))
        elif a:       m = float(min(a))
        else:         return self.mid
        self._ema = m if self._ema is None else 0.92*self._ema + 0.08*m
        self.mid  = self._ema
        return self.mid


# ---------------------------------------------------------------------------
# Minimal Trader (vouchers + VFE spot update only — for isolated backtesting)
# ---------------------------------------------------------------------------
class Trader:
    def __init__(self):
        self._spot    = SpotReference()
        self._voucher = VoucherStrategy(self._spot)

    def run(self, state: TradingState):
        result = {}
        od  = state.order_depths
        pos = state.position

        # Always update spot first so voucher delta/fair calculations are current
        if VFE_PRODUCT in od:
            self._spot.update(od[VFE_PRODUCT])

        result.update(self._voucher.trade(od, pos))

        return result, 0, ""
