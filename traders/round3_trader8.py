from __future__ import annotations
from typing import Dict, List
from collections import deque
import math

from datamodel import Order, TradingState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SCALP_STRIKES = [5300, 5400]   # 5200 excluded: borderline ITM → systematic bias

VFE_LIMIT      = 200
HYDROGEL_LIMIT = 200
OPTION_LIMIT   = 25

# Hydrogel — EMA MM with inventory-skewed quoting (v17 improvement)
HYDRO_CLIP      = 30
HYDRO_SKEW      = 0.10
HYDRO_TAKE_EDGE = 2
HYDRO_EMA_ALPHA = 0.15
HYDRO_HEAVY_POS = 100   # widen quotes by 2 ticks when |pos| > this

# VFE passive MM (unchanged from 408194)
VFE_MM_CLIP  = 15
VFE_MM_SKEW  = 0.06
# VFE mean reversion — raised threshold to 12 to filter noise
VFE_MR_EMA_ALPHA = 0.15
VFE_MR_THRESH    = 12    # was 8 in v17; 12 ≈ 2.4x half-spread, clears noise
VFE_MR_SIZE      = 20

# Options desk — flat per-strike TV deque (proven approach from 408194/trader7)
TV_DEQUE_LEN = 500
ORDER_SIZE   = 10

# TV seeds — Day 0 actuals for cold-start stability
TV_SEED: Dict[int, float] = {
    4000:  0.0,
    4500:  0.0,
    5000:  6.8,
    5100: 21.6,
    5200: 51.0,
    5300: 48.9,
    5400: 18.5,
    5500:  8.1,
    6000:  0.5,
    6500:  0.5,
}

# Entry thresholds — empirically validated p90 residuals from 3-day data
# Must exceed full bid-ask spread to guarantee positive EV per trade
# K=5300: spread~2.1, p90 resid=6.65 → net edge ≈ 4.5/trade
# K=5400: spread~1.4, p90 resid=3.17 → net edge ≈ 1.8/trade
ENTRY_THRESH: Dict[int, float] = {
    5300: 6.5,
    5400: 3.2,
}


# ---------------------------------------------------------------------------
# HYDROGEL_PACK — EMA market maker with heavy-position quote widening
# ---------------------------------------------------------------------------

class HydrogelStrategy:
    PRODUCT      = "HYDROGEL_PACK"
    DEFAULT_FAIR = 9990.0

    def __init__(self) -> None:
        self._fair = self.DEFAULT_FAIR

    def _update(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:
            mid = (max(b) + min(a)) / 2.0
        elif b:
            mid = float(max(b))
        elif a:
            mid = float(min(a))
        else:
            return self._fair
        self._fair = (1 - HYDRO_EMA_ALPHA) * self._fair + HYDRO_EMA_ALPHA * mid
        return self._fair

    def trade(self, od, pos: int) -> List[Order]:
        fair   = self._update(od)
        orders: List[Order] = []
        lim    = HYDROGEL_LIMIT

        # Aggressive taker: lift cheap asks, hit rich bids
        for ask in sorted(od.sell_orders):
            if ask <= fair - HYDRO_TAKE_EDGE and pos < lim:
                q = min(-od.sell_orders[ask], lim - pos)
                orders.append(Order(self.PRODUCT, ask, q))
                pos += q

        for bid in sorted(od.buy_orders, reverse=True):
            if bid >= fair + HYDRO_TAKE_EDGE and pos > -lim:
                q = min(od.buy_orders[bid], lim + pos)
                orders.append(Order(self.PRODUCT, bid, -q))
                pos -= q

        # Passive maker: inventory-skewed quotes
        skew   = int(round(HYDRO_SKEW * pos))
        bb     = max(od.buy_orders)  if od.buy_orders  else int(fair) - 8
        ba     = min(od.sell_orders) if od.sell_orders else int(fair) + 8
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)

        # Widen quotes when heavily long/short to slow inventory accumulation
        if pos >  HYDRO_HEAVY_POS:
            ask_px += 2
        if pos < -HYDRO_HEAVY_POS:
            bid_px -= 2

        if bid_px >= ask_px:
            bid_px = int(math.floor(fair)) - 1
            ask_px = int(math.ceil(fair))  + 1

        bq = min(HYDRO_CLIP, max(0, lim - pos))
        aq = min(HYDRO_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders


# ---------------------------------------------------------------------------
# VELVETFRUIT_EXTRACT — passive MM + lightweight mean reversion
# ---------------------------------------------------------------------------

class VFEStrategy:
    PRODUCT     = "VELVETFRUIT_EXTRACT"
    DEFAULT_MID = 5250.0

    def __init__(self) -> None:
        self._ema    = self.DEFAULT_MID   # slow EMA for passive MM (α=0.10)
        self._mr_ema = self.DEFAULT_MID   # fast EMA for mean reversion (α=0.15)

    def mid(self) -> float:
        return self._ema

    def update_ema(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:
            m = (max(b) + min(a)) / 2.0
        elif b:
            m = float(max(b))
        elif a:
            m = float(min(a))
        else:
            return self._ema
        self._ema    = 0.90 * self._ema    + 0.10 * m
        self._mr_ema = (1 - VFE_MR_EMA_ALPHA) * self._mr_ema + VFE_MR_EMA_ALPHA * m
        return self._ema

    def trade(self, od, pos: int) -> List[Order]:
        fair   = self.update_ema(od)
        orders: List[Order] = []
        lim    = VFE_LIMIT

        # Mean reversion taker (higher threshold = 12 to filter noise)
        if od.buy_orders and od.sell_orders:
            best_bid  = max(od.buy_orders)
            best_ask  = min(od.sell_orders)
            deviation = ((best_bid + best_ask) / 2.0) - self._mr_ema

            if deviation > VFE_MR_THRESH and pos > -lim:
                qty = min(od.buy_orders[best_bid], VFE_MR_SIZE, lim + pos)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_bid, -qty))
                    pos -= qty

            elif deviation < -VFE_MR_THRESH and pos < lim:
                qty = min(-od.sell_orders[best_ask], VFE_MR_SIZE, lim - pos)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_ask, qty))
                    pos += qty

        # Passive MM
        skew   = int(round(VFE_MM_SKEW * pos))
        bb     = max(od.buy_orders)  if od.buy_orders  else int(fair) - 2
        ba     = min(od.sell_orders) if od.sell_orders else int(fair) + 2
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)
        if bid_px >= ask_px:
            bid_px = int(math.floor(fair)) - 1
            ask_px = int(math.ceil(fair))  + 1
        if bid_px >= ask_px:
            return orders

        bq = min(VFE_MM_CLIP, max(0, lim - pos))
        aq = min(VFE_MM_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders


# ---------------------------------------------------------------------------
# Options Desk — proven flat-deque TV baseline (408194/trader7 approach)
# ---------------------------------------------------------------------------

class OptionsDeskStrategy:
    """
    Per-strike flat deque baseline for TV fair value.

    Why NOT quadratic smile fitting:
    - K=5200 is borderline ITM (S~5250); the parabola anchored on OTM strikes
      systematically underestimates its TV, generating false "cheap" signals.
    - With only 4-5 liquid belly strikes, a 3-parameter quadratic is overfit
      and oscillates before accumulating enough ticks (first 50) to be stable.
    - The flat deque per strike already captures the mean TV level correctly
      and the p90 threshold (6.5/3.2) filters transient noise without needing
      cross-strike detrending.

    Validated PnL: 5300=+2591, 5400=+1563 (trader7 backtest, 3-day total).
    """

    def __init__(self, vfe: VFEStrategy) -> None:
        self.vfe = vfe
        self._tv_deque: Dict[int, deque] = {
            K: deque([TV_SEED.get(K, 0.5)] * 100, maxlen=TV_DEQUE_LEN)
            for K in ALL_STRIKES
        }

    def trade(
        self,
        order_depths: Dict,
        positions:    Dict[str, int],
    ) -> Dict[str, List[Order]]:

        result: Dict[str, List[Order]] = {}
        S = self.vfe.mid()
        if S <= 0:
            return result

        for K in SCALP_STRIKES:
            product = f"VEV_{K}"
            od = order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            best_bid  = max(od.buy_orders)
            best_ask  = min(od.sell_orders)
            mid_opt   = (best_bid + best_ask) / 2.0
            intrinsic = max(0.0, S - K)
            tv_obs    = mid_opt - intrinsic

            if tv_obs >= 0:
                self._tv_deque[K].append(tv_obs)

            dq          = self._tv_deque[K]
            tv_baseline = sum(dq) / len(dq)
            fair        = intrinsic + tv_baseline
            thresh      = ENTRY_THRESH.get(K, 5.0)
            cur_pos     = positions.get(product, 0)
            orders: List[Order] = []

            # Buy when ask is deeply below fair TV baseline
            if cur_pos < OPTION_LIMIT and best_ask <= fair - thresh:
                qty = min(-od.sell_orders[best_ask], OPTION_LIMIT - cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

            # Sell when bid is deeply above fair TV baseline
            if cur_pos > -OPTION_LIMIT and best_bid >= fair + thresh:
                qty = min(od.buy_orders[best_bid], OPTION_LIMIT + cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                result[product] = orders

        return result


# ---------------------------------------------------------------------------
# Trader — top-level entry point
# ---------------------------------------------------------------------------

class Trader:
    def __init__(self) -> None:
        self.hydrogel = HydrogelStrategy()
        self.vfe      = VFEStrategy()
        self.options  = OptionsDeskStrategy(vfe=self.vfe)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        od  = state.order_depths
        pos = state.position

        # 1. Update VFE EMAs first — options desk reads spot price
        if "VELVETFRUIT_EXTRACT" in od:
            self.vfe.update_ema(od["VELVETFRUIT_EXTRACT"])

        # 2. Options desk: flat-deque TV taker
        result.update(self.options.trade(od, pos))

        # 3. VFE: mean reversion takes + passive MM
        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0),
            )

        # 4. Hydrogel: EMA market maker with heavy-pos quote widening
        if "HYDROGEL_PACK" in od:
            result["HYDROGEL_PACK"] = self.hydrogel.trade(
                od["HYDROGEL_PACK"],
                pos.get("HYDROGEL_PACK", 0),
            )

        return result, 0, ""
