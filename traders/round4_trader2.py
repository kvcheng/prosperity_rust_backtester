from __future__ import annotations
from typing import Dict, List
from collections import deque
import math

from datamodel import Order, TradingState

# ---------------------------------------------------------------------------
# Product constants
# ---------------------------------------------------------------------------
ALL_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SCALP_STRIKES = [5300, 5400]   # voucher strikes targeted for edge-harvesting

VFE_LIMIT     = 200
HYDROGEL_LIMIT = 200
OPTION_LIMIT  = 300

# ── Hydrogel tuning ─────────────────────────────────────────────────────────
HYDRO_EMA_ALPHA   = 0.15
HYDRO_TAKE_EDGE   = 2        # min edge (ticks) required to aggress
HYDRO_SKEW        = 0.04     # price offset per unit of inventory
HYDRO_CLIP        = 30       # max qty per resting order
HYDRO_HEAVY_POS   = 150      # widen quotes by 2 ticks when |pos| > this
HYDRO_SOFT_LONG   = 160
HYDRO_SOFT_SHORT  = -120
HYDRO_MAX_POST    = 30
HYDRO_HALF_SPREAD = 4        # fallback half-spread when book is thin

# ── VFE tuning ───────────────────────────────────────────────────────────────
VFE_MM_CLIP  = 15
VFE_MM_SKEW  = 0.06

# ── Options tuning ───────────────────────────────────────────────────────────
TV_DEQUE_LEN = 500
ORDER_SIZE   = 10

# Time-value seeds from historical data analysis
TV_SEED = {
    4000: 0.0,
    4500: 0.0,
    5000: 6.8,
    5100: 21.6,
    5200: 46.7,
    5300: 46.1,
    5400: 18.5,
    5500: 8.1,
    6000: 0.5,
    6500: 0.5,
}

# Minimum edge (in time-value units) required to trade each strike
ENTRY_THRESH = {
    5300: 6.5,
    5400: 3.2,
}


# ===========================================================================
# HYDROGEL_PACK  –  EMA fair value + trend slope, no counterparty signals
# ===========================================================================
class HydrogelStrategy:
    PRODUCT = "HYDROGEL_PACK"
    DEFAULT_FAIR = 9990.0

    def __init__(self):
        self.mid_history  = deque(maxlen=40)
        self.time_history = deque(maxlen=40)
        self.ema_mid      = None
        self.move_ema     = 1.0
        self.spread_ema   = None
        self.last_mid     = None

    # ── internal helpers ────────────────────────────────────────────────────
    def _observe_mid(self, od):
        b, a = od.buy_orders, od.sell_orders
        if b and a:   return (max(b) + min(a)) / 2.0
        if b:         return float(max(b))
        if a:         return float(min(a))
        return None

    def _update(self, od, timestamp: int):
        mid = self._observe_mid(od)
        if od.buy_orders and od.sell_orders:
            sp = min(od.sell_orders) - max(od.buy_orders)
            self.spread_ema = (float(sp) if self.spread_ema is None
                               else 0.85 * self.spread_ema + 0.15 * sp)
        if mid is not None:
            self.time_history.append(timestamp / 100.0)
            self.mid_history.append(mid)
            if self.last_mid is not None:
                mv = abs(mid - self.last_mid)
                self.move_ema = 0.85 * self.move_ema + 0.15 * mv
            self.last_mid = mid
            if self.ema_mid is None:
                self.ema_mid = mid
        return mid

    def _dynamic_alpha(self):
        sp_term = (0.0 if self.spread_ema is None
                   else min(0.10, self.spread_ema / 100.0))
        mv_term = min(0.12, self.move_ema / 50.0)
        return max(0.05, min(0.30, 0.05 + sp_term + mv_term))

    def _trend_slope(self):
        n = len(self.mid_history)
        if n < 6: return 0.0
        xs, ys = list(self.time_history), list(self.mid_history)
        mx, my = sum(xs) / n, sum(ys) / n
        d = sum((x - mx) ** 2 for x in xs)
        return 0.0 if d == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / d

    def fair_value(self, od, timestamp: int) -> float:
        mid = self._update(od, timestamp)
        if self.ema_mid is None:
            self.ema_mid = mid if mid is not None else self.DEFAULT_FAIR
        if mid is not None:
            alpha = self._dynamic_alpha()
            self.ema_mid = (1 - alpha) * self.ema_mid + alpha * mid
        trend      = self._trend_slope()
        trend_fair = self.ema_mid + trend * 3.0
        return (0.7 * self.ema_mid + 0.3 * trend_fair) if mid is not None else trend_fair

    # ── take step ───────────────────────────────────────────────────────────
    def _take(self, od, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        fair  = self.fair_value(od, timestamp)
        lim   = HYDROGEL_LIMIT
        edge  = 0.5
        if self.spread_ema is not None:
            edge += min(1.5, self.spread_ema / 20.0)
        edge += min(1.0, self.move_ema / 10.0)
        buy_edge  = edge + (0.5 if position > 60  else 0.0)
        sell_edge = edge + (0.5 if position <  0  else 0.0)

        for ask in sorted(od.sell_orders):
            vol = -od.sell_orders[ask]; qty = 0
            if ask <= fair - buy_edge  and position < lim:  qty = min(vol, lim - position)
            elif ask <= fair           and position < 0:     qty = min(vol, abs(position))
            if qty > 0:
                orders.append(Order(self.PRODUCT, ask, qty)); position += qty
        for bid in sorted(od.buy_orders, reverse=True):
            vol = od.buy_orders[bid]; qty = 0
            if bid >= fair + sell_edge and position > -lim: qty = min(vol, lim + position)
            elif bid >= fair           and position > 0:    qty = min(vol, position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bid, -qty)); position -= qty
        return orders

    # ── make step ───────────────────────────────────────────────────────────
    def _make(self, od, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        fair = self.fair_value(od, timestamp)
        lim  = HYDROGEL_LIMIT

        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None

        if bb is not None and ba is not None and ba > bb:
            half = max(1, (ba - bb) // 2)
        elif self.spread_ema is not None:
            half = max(1, int(round(self.spread_ema / 2)))
        else:
            half = HYDRO_HALF_SPREAD

        skew = HYDRO_SKEW * position
        bid_px = int(round(fair - half - skew))
        ask_px = int(round(fair + half - skew))

        if bb is not None:
            bid_px = max(bid_px, bb + 1 if bb < fair else bb)
        if ba is not None:
            ask_px = min(ask_px, ba - 1 if ba > fair else ba)
        if bid_px >= ask_px:
            bid_px = int(math.floor(fair)) - 1
            ask_px = int(math.ceil(fair))  + 1

        bq = min(HYDRO_MAX_POST, max(0, lim - position))
        aq = min(HYDRO_MAX_POST, max(0, lim + position))

        # Soft inventory limits: lean aggressively to flatten
        if position >= HYDRO_SOFT_LONG:
            ask_px  = min(ask_px, int(round(fair)))
            aq      = min(max(1, position), HYDRO_MAX_POST + 20)
            bq      = min(bq, 4)
        elif position <= HYDRO_SOFT_SHORT:
            bid_px  = max(bid_px, int(round(fair)))
            bq      = min(max(1, abs(position)), HYDRO_MAX_POST + 20)
            aq      = min(aq, 4)

        # Widen quotes when carrying heavy inventory
        if position > HYDRO_HEAVY_POS:  ask_px += 2
        if position < -HYDRO_HEAVY_POS: bid_px -= 2

        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders

    def trade(self, od, pos: int, timestamp: int) -> List[Order]:
        take  = self._take(od, pos, timestamp)
        pos  += sum(o.quantity for o in take)
        return take + self._make(od, pos, timestamp)


# ===========================================================================
# VELVETFRUIT_EXTRACT  –  slow-EMA passive market maker (from round3_trader9)
# ===========================================================================
class VFEStrategy:
    PRODUCT     = "VELVETFRUIT_EXTRACT"
    DEFAULT_MID = 5250.0

    def __init__(self) -> None:
        self._ema = self.DEFAULT_MID

    def mid(self) -> float:
        return self._ema

    def update_ema(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:   m = (max(b) + min(a)) / 2.0
        elif b:       m = float(max(b))
        elif a:       m = float(min(a))
        else:         return self._ema
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
        if bid_px >= ask_px:
            return orders

        bq = min(VFE_MM_CLIP, max(0, lim - pos))
        aq = min(VFE_MM_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders


# ===========================================================================
# VELVETFRUIT_EXTRACT_VOUCHER  –  time-value mean reversion (from round3_trader9)
# ===========================================================================
class OptionsDeskStrategy:
    def __init__(self, vfe: VFEStrategy):
        self.vfe = vfe
        self._tv_deque: Dict[int, deque] = {
            K: deque([TV_SEED.get(K, 0.5)] * 100, maxlen=TV_DEQUE_LEN)
            for K in ALL_STRIKES
        }

    def trade(self, order_depths, positions) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        S = self.vfe.mid()
        if S <= 0:
            return result

        for K in SCALP_STRIKES:
            product = f"VEV_{K}"
            od      = order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid_opt  = (best_bid + best_ask) / 2.0
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

            if cur_pos < OPTION_LIMIT and best_ask <= fair - thresh:
                qty = min(-od.sell_orders[best_ask], OPTION_LIMIT - cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

            if cur_pos > -OPTION_LIMIT and best_bid >= fair + thresh:
                qty = min(od.buy_orders[best_bid], OPTION_LIMIT + cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                result[product] = orders

        return result


# ===========================================================================
# Trader entry point
# ===========================================================================
class Trader:
    def __init__(self):
        self.hydrogel = HydrogelStrategy()
        self.vfe      = VFEStrategy()
        self.options  = OptionsDeskStrategy(vfe=self.vfe)

    def run(self, state: TradingState):
        result = {}
        od  = state.order_depths
        pos = state.position

        # # VFE EMA must be updated before options desk reads vfe.mid()
        if "VELVETFRUIT_EXTRACT" in od:
            self.vfe.update_ema(od["VELVETFRUIT_EXTRACT"])

        # # # Options desk
        # # result.update(self.options.trade(od, pos))

        # # # VFE market making
        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0),
            )

        # # Hydrogel market making
        # # if "HYDROGEL_PACK" in od:
        # #     result["HYDROGEL_PACK"] = self.hydrogel.trade(
        # #         od["HYDROGEL_PACK"],
        # #         pos.get("HYDROGEL_PACK", 0),
        # #         state.timestamp,
        # #     )

        return result, 0, ""
