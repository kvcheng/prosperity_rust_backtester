from __future__ import annotations
from typing import Dict, List
from collections import deque
import math
from datamodel import Order, TradingState

# ── Product constants ────────────────────────────────────────────────────────
ALL_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SCALP_STRIKES = [5300, 5000, 4000, 4500, 5100, 5200]
VFE_LIMIT     = 200
HYDRO_LIMIT   = 200
OPTION_LIMIT  = 300

# ── Hydrogel tuning ───────────────────────────────────────────────────────────
HYDRO_MAXPOST   = 100
HYDRO_HALFSPREAD = 1
HYDRO_SKEW      = 0.015        # inventory skew per unit
HYDRO_HEAVY     = 195
HYDRO_SOFTLONG  = 200
HYDRO_SOFTSHORT = -200

# KEY: ratio of HYDROGEL to VFE, learned from data (1.9046 ± 0.01)
HYDRO_VFE_RATIO_INIT = 1.9046
HYDRO_VFE_RATIO_ALPHA = 0.0002  # very slow update — ratio is highly persistent

# ── VFE tuning ────────────────────────────────────────────────────────────────
VFE_MM_CLIP  = 15
VFE_MM_SKEW  = 0.06

# Mark 67 informed-buy signal weight for VFE fair nudge
VFE_M67_SIGNAL_ALPHA = 0.85    # EMA decay for signal
VFE_M67_NUDGE_CAP    = 3.0     # max pts to nudge VFE fair up

# ── Options tuning ────────────────────────────────────────────────────────────
TV_DEQUE_LEN = 500
ORDER_SIZE   = 10
TV_SEED = {
    4000: 0.0, 4500: 0.0, 5000: 6.8, 5100: 21.6, 5200: 35.0,
    5300: 46.1, 5400: 18.5, 5500: 8.1, 6000: 0.5, 6500: 0.5,
}
ENTRY_THRESH = {
    5300: 4.75, 5000: 6.0, 4000: 0.5,
    4500: 0.0,  5100: 15.0, 5200: 15.25,
}

f = lambda K: f"VEV_{K}"


# ════════════════════════════════════════════════════════════════════════════════
#  VFEStrategy — with Mark 67 / Mark 55 counterparty signal
# ════════════════════════════════════════════════════════════════════════════════
class VFEStrategy:
    PRODUCT = "VELVETFRUIT_EXTRACT"
    DEFAULT  = 5250.0

    def __init__(self):
        self.ema    = self.DEFAULT
        # Mark 67: buyer-only, predicts +2pt rise in VFE
        # Mark 55: most active noise trader, prolific crosser
        # Signal: EMA of above-mid buy aggression (Mark 67 signature)
        self._m67_signal = 0.0   # informed buy signal (positive = bullish)
        self._m55_signal = 0.0   # noise sell signal (fades quickly)

    def mid(self) -> float:
        return self.ema

    def update_ema(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:   m = (max(b) + min(a)) / 2.0
        elif b:       m = float(max(b))
        elif a:       m = float(min(a))
        else:         return self.ema
        self.ema = 0.90 * self.ema + 0.10 * m
        return self.ema

    def update_cp(self, market_trades) -> None:
        """
        Mark 67: buyer-only, above-mid buys predict +2pt VFE rise.
        Mark 55: noise crosser — large volume but negative edge.
        Mark 49: selling by Mark 49 is also a bullish signal for VFE.
        """
        m67_raw = 0.0
        m55_raw = 0.0
        for tr in market_trades:
            q = abs(int(tr.quantity))
            if tr.buyer  == "Mark 67": m67_raw += q        # bullish
            if tr.buyer  == "Mark 49": m67_raw += 0.5 * q  # mild bullish (Mark 49 sell = bullish)
            if tr.seller == "Mark 55": m55_raw += q        # noise
            if tr.buyer  == "Mark 55": m55_raw += q        # noise
        # Informed signal decays slowly; noise decays quickly
        self._m67_signal = VFE_M67_SIGNAL_ALPHA * self._m67_signal + (1-VFE_M67_SIGNAL_ALPHA) * m67_raw
        self._m55_signal = 0.70 * self._m55_signal + 0.30 * m55_raw

    def fair_value(self) -> float:
        """
        Base fair = EMA.
        Nudge up when Mark 67 informed-buy signal is active.
        Mark 55 noise does NOT affect fair — it's just noise we trade against.
        """
        nudge = min(VFE_M67_NUDGE_CAP, 0.08 * self._m67_signal)
        return self.ema + nudge

    def trade(self, od, pos: int) -> List[Order]:
        fair = self.fair_value()
        orders = []
        lim  = VFE_LIMIT
        skew = int(round(VFE_MM_SKEW * pos))
        bb   = max(od.buy_orders)  if od.buy_orders  else int(fair - 2)
        ba   = min(od.sell_orders) if od.sell_orders else int(fair + 2)
        bidpx = min(bb + 1 - skew, int(math.floor(fair - 1)))
        askpx = max(ba - 1 - skew, int(math.ceil(fair  + 1)))
        if bidpx >= askpx:
            bidpx = int(math.floor(fair - 1))
            askpx = int(math.ceil(fair  + 1))
        if bidpx >= askpx:
            return orders
        bq = min(VFE_MM_CLIP, max(0, lim - pos))
        aq = min(VFE_MM_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bidpx,  bq))
        if aq: orders.append(Order(self.PRODUCT, askpx, -aq))
        return orders


# ════════════════════════════════════════════════════════════════════════════════
#  HydrogelStrategy — VFE-anchored fair value + Mark 38 counterparty signal
# ════════════════════════════════════════════════════════════════════════════════
class HydrogelStrategy:
    PRODUCT     = "HYDROGEL_PACK"
    DEFAULT_FAIR = 9990.0

    def __init__(self, vfe: VFEStrategy):
        self.vfe          = vfe
        # Ratio tracker: HYDROGEL / VFE (learned from data ≈ 1.9046)
        self._ratio_ema   = HYDRO_VFE_RATIO_INIT
        # Standalone EMA for when VFE data is unavailable
        self._ema_mid     = None
        self._spread_ema  = None
        self._move_ema    = 1.0
        self._last_mid    = None
        # Counterparty signal (Mark 38 = noise, Mark 14 = smart)
        self.cp_signal    = 0.0

    def _observe_mid(self, od):
        b, a = od.buy_orders, od.sell_orders
        if b and a: return (max(b) + min(a)) / 2.0
        if b:       return float(max(b))
        if a:       return float(min(a))
        return None

    def update_cp(self, market_trades) -> None:
        raw = 0.0
        for tr in market_trades:
            q = abs(int(tr.quantity))
            if tr.buyer  == "Mark 14": raw += q
            if tr.seller == "Mark 14": raw -= q
            if tr.buyer  == "Mark 38": raw -= 0.8 * q   # Mark 38 buys = fade it
            if tr.seller == "Mark 38": raw += 0.8 * q   # Mark 38 sells = fade it
        self.cp_signal = 0.70 * self.cp_signal + raw

    def _update(self, od, timestamp: int):
        mid = self._observe_mid(od)
        if od.buy_orders and od.sell_orders:
            sp = min(od.sell_orders) - max(od.buy_orders)
            self._spread_ema = float(sp) if self._spread_ema is None else 0.85*self._spread_ema + 0.15*sp
        if mid is not None:
            if self._last_mid is not None:
                self._move_ema = 0.85*self._move_ema + 0.15*abs(mid - self._last_mid)
            self._last_mid = mid
            if self._ema_mid is None:
                self._ema_mid = mid
        # Update ratio with latest observation
        vfe_mid = self.vfe.mid()
        if mid is not None and vfe_mid > 0:
            obs_ratio = mid / vfe_mid
            self._ratio_ema = ((1 - HYDRO_VFE_RATIO_ALPHA) * self._ratio_ema
                               + HYDRO_VFE_RATIO_ALPHA * obs_ratio)
        return mid

    def _dynamic_alpha(self):
        spterm = 0.0 if self._spread_ema is None else min(0.10, self._spread_ema/100.0)
        mvterm = min(0.12, self._move_ema / 50.0)
        return max(0.05, min(0.30, 0.05 + spterm + mvterm))

    def fair_value(self, od, timestamp: int) -> float:
        mid = self._update(od, timestamp)
        vfe_mid = self.vfe.mid()

        # PRIMARY: VFE-anchored fair value
        vfe_fair = self._ratio_ema * vfe_mid

        # SECONDARY: standalone EMA (blended in slowly)
        if self._ema_mid is None:
            self._ema_mid = mid or vfe_fair
        if mid is not None:
            alpha = self._dynamic_alpha()
            self._ema_mid = (1-alpha)*self._ema_mid + alpha*mid

        # Blend: 80% VFE-anchored (predictive), 20% own EMA (local)
        base_fair = 0.80 * vfe_fair + 0.20 * self._ema_mid

        # Counterparty signal adjustment
        cp_clip = max(-12.0, min(12.0, self.cp_signal))
        cp_adj  = 0.08 * cp_clip

        return base_fair + cp_adj

    def take(self, od, position: int, timestamp: int) -> List[Order]:
        orders = []
        fair   = self.fair_value(od, timestamp)
        lim    = HYDRO_LIMIT
        edge   = 0.5
        if self._spread_ema is not None:
            edge = min(1.5, self._spread_ema / 20.0)
        edge += min(1.0, self._move_ema / 10.0)
        cp_clip  = max(-12.0, min(12.0, self.cp_signal))
        buyedge  = edge + (0.5 if position > 60 else 0.0)
        selledge = edge + (0.5 if position < 0  else 0.0)
        # When Mark 38 is selling (cp_signal positive), be more aggressive buying
        if cp_clip < -8: selledge -= 0.35; buyedge += 0.40
        elif cp_clip > 8: buyedge -= 0.15

        for ask in sorted(od.sell_orders):
            vol = -od.sell_orders[ask]
            qty = 0
            if ask <= fair - buyedge and position < lim:
                qty = min(vol, lim - position)
            elif ask <= fair and position < 0:
                qty = min(vol, abs(position))
            if qty > 0:
                orders.append(Order(self.PRODUCT, ask, qty))
                position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            vol = od.buy_orders[bid]
            qty = 0
            if bid >= fair + selledge and position > -lim:
                qty = min(vol, lim + position)
            elif bid >= fair and position > 0:
                qty = min(vol, position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bid, -qty))
                position -= qty

        return orders

    def make(self, od, position: int, timestamp: int) -> List[Order]:
        orders  = []
        fair    = self.fair_value(od, timestamp)
        lim     = HYDRO_LIMIT
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None

        if bb is not None and ba is not None and ba > bb:
            half = max(1, (ba - bb) // 2)
        elif self._spread_ema is not None:
            half = max(1, int(round(self._spread_ema / 2)))
        else:
            half = HYDRO_HALFSPREAD

        cp_clip  = max(-12.0, min(12.0, self.cp_signal))
        cp_skew  = -0.02 * cp_clip
        skew     = HYDRO_SKEW * position + cp_skew

        bidpx = int(round(fair - half - skew))
        askpx = int(round(fair + half - skew))

        if bb is not None: bidpx = max(bidpx, bb + 1) if bb < fair else bb
        if ba is not None: askpx = min(askpx, ba - 1) if ba > fair else ba
        if bidpx >= askpx:
            bidpx = int(math.floor(fair - 1))
            askpx = int(math.ceil(fair  + 1))

        bq = min(HYDRO_MAXPOST, max(0, lim - position))
        aq = min(HYDRO_MAXPOST, max(0, lim + position))

        if position >= HYDRO_SOFTLONG:
            askpx = min(askpx, int(round(fair)))
            aq    = min(max(1, position, HYDRO_MAXPOST - 20), HYDRO_MAXPOST)
            bq    = min(bq, 4)
        elif position <= HYDRO_SOFTSHORT:
            bidpx = max(bidpx, int(round(fair)))
            bq    = min(max(1, abs(position), HYDRO_MAXPOST - 20), HYDRO_MAXPOST)
            aq    = min(aq, 4)

        if position >=  HYDRO_HEAVY: askpx += 2
        if position <= -HYDRO_HEAVY: bidpx -= 2

        # When Mark 38 is buying heavily (cp_signal strongly negative), be a better seller
        if cp_clip < -10:
            bq = min(bq, 6)
            aq = min(HYDRO_MAXPOST - 20, max(aq, 20))

        if bq: orders.append(Order(self.PRODUCT, bidpx,  bq))
        if aq: orders.append(Order(self.PRODUCT, askpx, -aq))
        return orders

    def trade(self, od, pos: int, timestamp: int, market_trades) -> List[Order]:
        self.update_cp(market_trades)
        take = self.take(od, pos, timestamp)
        pos += sum(o.quantity for o in take)
        return take + self.make(od, pos, timestamp)


# ════════════════════════════════════════════════════════════════════════════════
#  OptionsDeskStrategy — unchanged structure, kept for completeness
# ════════════════════════════════════════════════════════════════════════════════
class OptionsDeskStrategy:
    def __init__(self, vfe: VFEStrategy):
        self.vfe = vfe
        self.tv_deque: Dict[int, deque] = {
            K: deque([TV_SEED.get(K, 0.5)] * 100, maxlen=TV_DEQUE_LEN)
            for K in ALL_STRIKES
        }

    def trade(self, order_depths, positions) -> Dict[str, List[Order]]:
        result = {}
        S = self.vfe.mid()
        if S <= 0: return result
        for K in SCALP_STRIKES:
            product = f(K)
            od = order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders: continue
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid_opt  = (best_bid + best_ask) / 2.0
            intrinsic = max(0.0, S - K)
            tv_obs    = mid_opt - intrinsic
            if tv_obs >= 0:
                self.tv_deque[K].append(tv_obs)
            dq         = self.tv_deque[K]
            tv_baseline = sum(dq) / len(dq)
            fair       = intrinsic + tv_baseline
            thresh     = ENTRY_THRESH.get(K, 5.0)
            cur_pos    = positions.get(product, 0)
            orders     = []
            if cur_pos < OPTION_LIMIT and best_ask <= fair - thresh:
                qty = min(-od.sell_orders[best_ask], OPTION_LIMIT - cur_pos, ORDER_SIZE)
                if qty > 0: orders.append(Order(product, best_ask, qty))
            if cur_pos > -OPTION_LIMIT and best_bid >= fair + thresh:
                qty = min(od.buy_orders[best_bid], OPTION_LIMIT + cur_pos, ORDER_SIZE)
                if qty > 0: orders.append(Order(product, best_bid, -qty))
            if orders: result[product] = orders
        return result


# ════════════════════════════════════════════════════════════════════════════════
#  Trader
# ════════════════════════════════════════════════════════════════════════════════
class Trader:
    def __init__(self):
        self.vfe     = VFEStrategy()
        self.hydrogel = HydrogelStrategy(self.vfe)
        self.options  = OptionsDeskStrategy(self.vfe)

    def run(self, state: TradingState):
        result = {}
        od  = state.order_depths
        pos = state.position

        # 1. Update VFE EMA first — everything downstream reads vfe.mid()
        if "VELVETFRUIT_EXTRACT" in od:
            self.vfe.update_ema(od["VELVETFRUIT_EXTRACT"])
            # Update VFE counterparty signal from market trades
            vfe_trades = state.market_trades.get("VELVETFRUIT_EXTRACT", [])
            if vfe_trades:
                self.vfe.update_cp(vfe_trades)

        # 2. Options desk (reads VFE mid)
        result.update(self.options.trade(od, pos))

        # 3. VFE market making
        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"], pos.get("VELVETFRUIT_EXTRACT", 0))

        # 4. HYDROGEL — anchored to VFE, with Mark 38 signal
        if "HYDROGEL_PACK" in od:
            result["HYDROGEL_PACK"] = self.hydrogel.trade(
                od["HYDROGEL_PACK"],
                pos.get("HYDROGEL_PACK", 0),
                state.timestamp,
                state.market_trades.get("HYDROGEL_PACK", []))

        return result, 0, ""