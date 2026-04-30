"""
Round 3 — Empirical TV-EMA Scalping Trader  (v11)
==================================================

FIXES vs v10:
  1. Hydrogel: v3-identical single EMA (α=0.15) for both quoting AND taking.
  2. VFE: pure passive MM, zero delta hedging.
  3. TV EMA seeds: Day 0 actuals (not 3-day averages).
  4. Day multiplier: REMOVED — EMA already tracks natural TV decay.
  5. VFE crossed-quotes guard: never post when our_spread <= 0.
  6. [NEW v11] Per-day EMA reset at day boundaries for K=5200/5100:
     α=0.005 accumulates a 10+ unit lag by Day 2 for high-TV strikes,
     inverting sell signals and locking in theta losses as "realized edge".
     Fix: hard-reset EMA to observed TV at each new day.
  7. [NEW v11] Faster per-strike α tuned to each strike's TV volatility:
     K=5400/5500 α=0.05  (low TV, small intraday moves)
     K=5300     α=0.06  (medium)
     K=5200/5100 α=0.08  (high TV, larger intraday swings)
  8. [NEW v11] Disable K=5100: only 2 signals per 10k ticks, 4.3-unit spread
     means cost > edge on nearly every fill.
  9. [NEW v11] Raise K=5200 threshold to 6.0 (> 1.9σ residual std=3.16)
     to avoid trading at 1.27σ deviations that don't reliably mean-revert.

STRATEGY:
  HYDROGEL_PACK       → Single-EMA skewed market maker (v3-identical, +13,755)
  VELVETFRUIT_EXTRACT → Pure passive MM, no hedging (+3,914 baseline)
  VEV_5400/5300/5200  → Per-day-anchored TV-EMA taker (TAKER ONLY)
                        Buy: ask < intrinsic + tv_ema - thresh
                        Sell: bid > intrinsic + tv_ema + thresh

EXPECTED PERFORMANCE (3-day):
  HYDROGEL_PACK:        ~+13,755  (proven, identical logic to v3)
  VELVETFRUIT_EXTRACT:  ~+6,000   (slight improvement from crossed-quotes guard)
  VEV_5400:             ~+1,500   (confirmed profitable in every version)
  VEV_5300:             ~+1,200   (profitable once EMA tracks correctly)
  VEV_5200:             ~+300     (marginal; day-reset prevents theta bleed)
  TOTAL:                ~+22,755
"""

from __future__ import annotations
from typing import Dict, List, Optional
import math

from datamodel import Order, TradingState


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

ALL_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Strikes we actively scalp — 5100 excluded (spread 4.3, near-zero signal rate)
SCALP_STRIKES = [5400, 5300, 5200, 5500]

VFE_LIMIT      = 200
HYDROGEL_LIMIT = 200
OPTION_LIMIT   = 25   # per strike — small enough to avoid expensive hedging

# ── Hydrogel ─────────────────────────────────────────────────────────────────
HYDRO_CLIP      = 30
HYDRO_SKEW      = 0.10
HYDRO_TAKE_EDGE = 2
HYDRO_EMA_ALPHA = 0.15   # single EMA, v3-identical

# ── VFE ──────────────────────────────────────────────────────────────────────
VFE_MM_CLIP = 15
VFE_MM_SKEW = 0.06

# ── Options: TV EMA ──────────────────────────────────────────────────────────
# Per-strike α — faster for high-TV strikes to track intraday decay
TV_EMA_ALPHA: Dict[int, float] = {
    5100: 0.08,
    5200: 0.08,
    5300: 0.06,
    5400: 0.05,
    5500: 0.05,
}
TV_EMA_ALPHA_DEFAULT = 0.05

# Entry thresholds: must exceed full spread to survive transaction cost.
# K=5200 raised to 6.0 (>1.9σ) — at 1.27σ deviations do not reliably revert.
# Spread reference: 5200=2.9, 5300=2.1, 5400=1.4, 5500=1.4
ENTRY_THRESH: Dict[int, float] = {
    5100: 99.0,   # effectively disabled
    5200: 6.0,
    5300: 3.0,
    5400: 2.0,
    5500: 2.0,
}

# TV seeds: Day 0 actuals (NOT 3-day averages)
# Using Day 0 values prevents cold-start mis-selling during first ~20 ticks.
TV_SEED: Dict[int, float] = {
    4000: 0.0,
    4500: 0.0,
    5000: 6.8,
    5100: 21.6,
    5200: 51.0,
    5300: 48.9,
    5400: 18.5,
    5500: 8.1,
    6000: 0.5,
    6500: 0.5,
}

TICKS_PER_DAY = 10_000


# ─────────────────────────────────────────────────────────────────────────────
#  HYDROGEL_PACK  —  v3-Identical Single-EMA Market Maker
# ─────────────────────────────────────────────────────────────────────────────

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

        # ── Aggressive taker: lift stale cheap asks / hit stale rich bids ──
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

        # ── Passive maker: quote inside spread, inventory-skewed ───────────
        skew   = int(round(HYDRO_SKEW * pos))
        bb     = max(od.buy_orders)  if od.buy_orders  else int(fair) - 8
        ba     = min(od.sell_orders) if od.sell_orders else int(fair) + 8
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)
        if bid_px >= ask_px:
            bid_px = int(math.floor(fair)) - 1
            ask_px = int(math.ceil(fair))  + 1

        bq = min(HYDRO_CLIP, max(0, lim - pos))
        aq = min(HYDRO_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders


# ─────────────────────────────────────────────────────────────────────────────
#  VELVETFRUIT_EXTRACT  —  Pure Passive MM (NO delta hedging)
# ─────────────────────────────────────────────────────────────────────────────

class VFEStrategy:
    PRODUCT     = "VELVETFRUIT_EXTRACT"
    DEFAULT_MID = 5250.0

    def __init__(self) -> None:
        self._ema = self.DEFAULT_MID

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
        self._ema = 0.90 * self._ema + 0.10 * m
        return self._ema

    def trade(self, od, pos: int) -> List[Order]:
        fair   = self.update_ema(od)
        orders: List[Order] = []
        lim    = VFE_LIMIT

        skew   = int(round(VFE_MM_SKEW * pos))
        bb     = max(od.buy_orders)  if od.buy_orders  else int(fair) - 2
        ba     = min(od.sell_orders) if od.sell_orders else int(fair) + 2
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)
        if bid_px >= ask_px:
            bid_px = int(math.floor(fair)) - 1
            ask_px = int(math.ceil(fair))  + 1

        # Guard: never post crossed or zero-spread quotes
        # (occurs when market spread = 1 or 2 and we step inside)
        if bid_px >= ask_px:
            return orders

        bq = min(VFE_MM_CLIP, max(0, lim - pos))
        aq = min(VFE_MM_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders


# ─────────────────────────────────────────────────────────────────────────────
#  Options Desk  —  Per-Day-Anchored TV-EMA Taker
# ─────────────────────────────────────────────────────────────────────────────

class OptionsDeskStrategy:
    """
    Per-strike TV EMA with per-day reset to eliminate cross-day theta bleed.

    WHY PER-DAY RESET:
        TV for K=5200 falls from ~51 (Day 0) to ~39 (Day 2) — a 12-unit drop.
        With α=0.005 (half-life ~139 ticks), the EMA accumulates a 10+ unit
        downward lag by Day 2. This causes fair-value estimates to be 10 units
        too low on Day 2, which inverts sell signals: we sell at prices that
        are actually BELOW true fair, locking in the theta loss as fake "edge".
        Resetting EMA to observed TV at each day boundary eliminates this.

    SIGNAL QUALITY FILTER (K=5200):
        TV residual std = 3.16. Old threshold of 4.0 = only 1.27σ, insufficient
        for consistent mean-reversion. New threshold = 6.0 (1.90σ).

    NO PASSIVE QUOTING:
        TV autocorrelation at lag-1 = 0.968–0.988 across all belly strikes.
        Every passive fill is adversely selected. Taker only.

    NO DELTA HEDGING:
        Max 25 contracts per strike × 0.5 delta = 12.5 units gross delta.
        Hedging at 2.5 half-spread × ~150 events/session costs more than
        total options PnL. Disabled entirely.
    """

    def __init__(self, vfe: VFEStrategy) -> None:
        self.vfe = vfe

        # EMA state per strike
        self._tv_ema:   Dict[int, float] = {K: TV_SEED.get(K, 0.5) for K in ALL_STRIKES}
        self._tv_obs:   Dict[int, int]   = {K: 0 for K in ALL_STRIKES}

        # Day-boundary tracking (one reset per strike per new day)
        self._last_day: Dict[int, int]   = {K: -1 for K in ALL_STRIKES}

    # ── EMA update with per-day reset ────────────────────────────────────────

    def _update_tv(self, K: int, tv_obs: float, day: int) -> None:
        """
        Update the TV EMA for strike K.
        Resets to observed TV at each new day to prevent cross-day drift.
        Uses faster α for high-TV strikes (5200/5100) to track intraday decay.
        """
        # Hard reset at day boundary
        if day != self._last_day[K]:
            if self._last_day[K] >= 0:
                # Re-anchor to current observed TV (not seed)
                self._tv_ema[K]  = tv_obs
                self._tv_obs[K]  = 0
            self._last_day[K] = day

        alpha = TV_EMA_ALPHA.get(K, TV_EMA_ALPHA_DEFAULT)

        if self._tv_obs[K] < 20:
            # Fast warmup blend: pull toward observed TV quickly
            self._tv_ema[K] = 0.20 * tv_obs + 0.80 * self._tv_ema[K]
        else:
            self._tv_ema[K] = (1 - alpha) * self._tv_ema[K] + alpha * tv_obs

        self._tv_obs[K] += 1

    def _fair(self, S: float, K: int) -> float:
        """Fair option price = intrinsic + EMA of time value. No day multiplier."""
        intrinsic = max(0.0, S - K)
        return intrinsic + max(0.0, self._tv_ema[K])

    # ── Main trade method ─────────────────────────────────────────────────────

    def trade(
        self,
        order_depths: Dict,
        positions:    Dict[str, int],
        timestamp:    int,
    ) -> Dict[str, List[Order]]:

        result: Dict[str, List[Order]] = {}

        S = self.vfe.mid()
        if S <= 0:
            return result

        # Day index: 0, 1, 2 — derived from timestamp
        day = int(timestamp // TICKS_PER_DAY)

        for K in SCALP_STRIKES:
            product = f"VEV_{K}"
            od = order_depths.get(product)
            if od is None:
                continue

            # ── Update TV EMA from current mid-price ─────────────────────────
            if od.buy_orders and od.sell_orders:
                best_bid  = max(od.buy_orders)
                best_ask  = min(od.sell_orders)
                mid_opt   = (best_bid + best_ask) / 2.0
                intrinsic = max(0.0, S - K)
                tv_obs    = mid_opt - intrinsic

                if tv_obs >= 0:
                    self._update_tv(K, tv_obs, day)
            else:
                # One-sided or empty book — skip trading this tick
                continue

            cur_pos = positions.get(product, 0)
            fair    = self._fair(S, K)
            thresh  = ENTRY_THRESH.get(K, 3.0)
            orders: List[Order] = []

            # ── Buy cheap ask ─────────────────────────────────────────────────
            if cur_pos < OPTION_LIMIT and od.sell_orders:
                best_ask = min(od.sell_orders)
                if best_ask <= fair - thresh:
                    qty = min(
                        -od.sell_orders[best_ask],
                        OPTION_LIMIT - cur_pos,
                        10,
                    )
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))

            # ── Sell rich bid ─────────────────────────────────────────────────
            if cur_pos > -OPTION_LIMIT and od.buy_orders:
                best_bid = max(od.buy_orders)
                if best_bid >= fair + thresh:
                    qty = min(
                        od.buy_orders[best_bid],
                        OPTION_LIMIT + cur_pos,
                        10,
                    )
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))

            if orders:
                result[product] = orders

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Trader  —  Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """
    Round 3 v11 — Per-Day-Anchored TV-EMA Scalper

    Products:
        HYDROGEL_PACK       → v3-identical single-EMA skewed market maker
        VELVETFRUIT_EXTRACT → pure passive MM, no hedging
        VEV_5400 / 5300 / 5200 / 5500 → per-day TV-EMA taker

    Change log vs v10:
        + Per-day EMA hard-reset at day boundaries (eliminates cross-day drift)
        + Faster per-strike α tuned to each strike's TV volatility
        + K=5200 threshold raised to 6.0 (1.90σ vs old 1.27σ)
        + K=5100 effectively disabled (threshold=99, near-zero signal rate)
        + Timestamp-derived day index (no external state needed)
    """

    def __init__(self) -> None:
        self.hydrogel = HydrogelStrategy()
        self.vfe      = VFEStrategy()
        self.options  = OptionsDeskStrategy(vfe=self.vfe)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        od  = state.order_depths
        pos = state.position
        ts  = state.timestamp

        # 1. Update VFE EMA first — options desk needs spot estimate
        if "VELVETFRUIT_EXTRACT" in od:
            self.vfe.update_ema(od["VELVETFRUIT_EXTRACT"])

        # 2. Options desk: update TV EMAs + fire taker orders
        result.update(self.options.trade(od, pos, ts))

        # 3. VFE: pure passive MM (overwrites any options orders for VFE)
        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0),
            )

        # 4. Hydrogel: independent v3-identical market maker
        if "HYDROGEL_PACK" in od:
            result["HYDROGEL_PACK"] = self.hydrogel.trade(
                od["HYDROGEL_PACK"],
                pos.get("HYDROGEL_PACK", 0),
            )

        return result, 0, ""
