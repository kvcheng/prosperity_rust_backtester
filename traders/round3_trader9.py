from __future__ import annotations
from typing import Dict, List
from collections import deque
import math

from datamodel import Order, TradingState


ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SCALP_STRIKES = [5300, 5400]  

VFE_LIMIT = 200
HYDROGEL_LIMIT = 200
OPTION_LIMIT = 200


HYDRO_CLIP = 30
HYDRO_SKEW = 0.10
HYDRO_TAKE_EDGE = 2
HYDRO_EMA_ALPHA = 0.15
HYDRO_HEAVY_POS = 100  


VFE_MM_CLIP = 15
VFE_MM_SKEW = 0.06


TV_DEQUE_LEN = 500
ORDER_SIZE = 10


TV_SEED = {
    4000: 0.0,
    4500: 0.0,
    5000: 6.8,
    5100: 21.6,
    5200: 51.0,
    5300: 46.1,
    5400: 18.5,
    5500: 8.1,
    6000: 0.5,
    6500: 0.5,
}


ENTRY_THRESH = {
    5300: 6.5,
    5400: 3.2,
}

class HydrogelStrategy:
    PRODUCT = "HYDROGEL_PACK"
    DEFAULT_FAIR = 9990.0

    def __init__(self):
        self._fair = self.DEFAULT_FAIR

    def _update(self, od):
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

    def trade(self, od, pos: int):
        fair = self._update(od)
        orders: List[Order] = []
        lim = HYDROGEL_LIMIT

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

        skew = int(round(HYDRO_SKEW * pos))
        bb = max(od.buy_orders)  if od.buy_orders  else int(fair) - 8
        ba = min(od.sell_orders) if od.sell_orders else int(fair) + 8
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)

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

class OptionsDeskStrategy:
    def __init__(self, vfe: VFEStrategy):
        self.vfe = vfe
        self._tv_deque: Dict[int, deque] = {
            K: deque([TV_SEED.get(K, 0.5)] * 100, maxlen=TV_DEQUE_LEN)
            for K in ALL_STRIKES
        }

    def trade(self, order_depths, positions):
        result = {}
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

class Trader:
    def __init__(self):
        self.hydrogel = HydrogelStrategy()
        self.vfe      = VFEStrategy()
        self.options  = OptionsDeskStrategy(vfe=self.vfe)

    def run(self, state):
        result = {}
        od  = state.order_depths
        pos = state.position

        if "VELVETFRUIT_EXTRACT" in od:
            self.vfe.update_ema(od["VELVETFRUIT_EXTRACT"])

        result.update(self.options.trade(od, pos))

        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0),
            )

        if "HYDROGEL_PACK" in od:
            result["HYDROGEL_PACK"] = self.hydrogel.trade(
                od["HYDROGEL_PACK"],
                pos.get("HYDROGEL_PACK", 0),
            )

        return result, 0, ""