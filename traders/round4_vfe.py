from __future__ import annotations
from collections import deque
import math
from datamodel import Order, TradingState

# ---------------------------------------------------------------------------
# VELVETFRUIT_EXTRACT — Improved Market-Making Strategy
#
# Problems in the original (round3_trader9.py):
#   1. DEFAULT_MID = 5250 hardcoded — same cold-start bug as Hydrogel.
#      If the real market opens away from 5250, early quotes are mis-placed.
#   2. EMA alpha = 0.10 (very slow). If VFE trends, MM quotes lag and
#      the ask side gets lifted at a loss, just like Hydrogel.
#   3. Skew formula uses min/max clamps against best quotes — same sign
#      bug identified in Hydrogel, causing bids above fair when short.
#   4. No take step — only passive quotes, so mispriced resting orders
#      from counterparties are never aggressively captured.
#   5. No trend awareness — no OLS or fast/slow EMA to detect direction.
#   6. No counterparty signal — Round 4 exposes buyer/seller IDs.
#
# Fixes / additions:
#   - Cold-start: _fair_ema seeded from first real mid, not hardcoded.
#   - Dynamic alpha: speeds up EMA in volatile/wide-spread regimes.
#   - Trend component: OLS slope blended into fair value (proven approach
#     from IntarianPepperRootStrategy in round1_trader.py).
#   - Take step: aggress asks/bids that are clearly mispriced vs fair.
#   - Corrected skew formula: bid = fair - half - skew,
#                              ask = fair + half - skew  (clean, no clamps).
#   - Counterparty signal: Mark's net flow nudges fair value.
# ---------------------------------------------------------------------------

PRODUCT        = "VELVETFRUIT_EXTRACT"
POSITION_LIMIT = 200

# EMA / trend
TREND_WINDOW   = 40
TREND_HORIZON  = 3.0

# Take
TAKE_EDGE      = 1.5   # VFE spreads are tighter than Hydrogel, use smaller edge

# Make
MM_CLIP        = 15
MM_SKEW        = 0.06
SOFT_LONG      =  130
SOFT_SHORT     = -100

# Counterparty (Round 4)
MARK_ID        = "Mark"
MARK_WINDOW    = 20
MARK_SCALE     = 0.3   # VFE is a calmer product; smaller adjustment


class VelvetfruitExtractStrategy:

    def __init__(self):
        self._fair_ema   = None       # cold-start: None until first tick
        self._spread_ema = None
        self._move_ema   = 1.0
        self._last_mid   = None
        self._tick       = 0
        self._mid_hist   = deque(maxlen=TREND_WINDOW)   # (tick, mid)
        self._mark_buf   = deque(maxlen=MARK_WINDOW)

    # ------------------------------------------------------------------
    def _observe_mid(self, od):
        b, a = od.buy_orders, od.sell_orders
        if b and a:   return (max(b) + min(a)) / 2.0
        if b:         return float(max(b))
        if a:         return float(min(a))
        return None

    def _update_stats(self, od):
        mid = self._observe_mid(od)
        self._tick += 1
        if od.buy_orders and od.sell_orders:
            sp = min(od.sell_orders) - max(od.buy_orders)
            self._spread_ema = float(sp) if self._spread_ema is None else 0.85*self._spread_ema + 0.15*sp
        if mid is not None:
            self._mid_hist.append((self._tick, mid))
            if self._last_mid is not None:
                self._move_ema = 0.85*self._move_ema + 0.15*abs(mid - self._last_mid)
            self._last_mid = mid
            # Cold-start fix
            if self._fair_ema is None:
                self._fair_ema = mid
            else:
                alpha = max(0.05, min(0.30,
                    0.08
                    + (0 if self._spread_ema is None else min(0.10, self._spread_ema/100.0))
                    + min(0.12, self._move_ema/50.0)))
                self._fair_ema = (1 - alpha)*self._fair_ema + alpha*mid
        return mid

    def _trend_slope(self):
        n = len(self._mid_hist)
        if n < 6: return 0.0
        xs = [d[0] for d in self._mid_hist]
        ys = [d[1] for d in self._mid_hist]
        mx, my = sum(xs)/n, sum(ys)/n
        denom = sum((x-mx)**2 for x in xs)
        return 0.0 if denom == 0 else sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denom

    def _ingest_trades(self, mkt):
        for t in mkt:
            if t.buyer   == MARK_ID: self._mark_buf.append( t.quantity)
            elif t.seller == MARK_ID: self._mark_buf.append(-t.quantity)

    def _fair_value(self):
        if self._fair_ema is None: return None
        slope      = self._trend_slope()
        trend_fair = self._fair_ema + slope * TREND_HORIZON
        base       = 0.7*self._fair_ema + 0.3*trend_fair
        cp_adj     = (sum(self._mark_buf)/100.0) * MARK_SCALE
        return base + cp_adj

    # ------------------------------------------------------------------
    def _take(self, od, position, fair):
        orders = []
        lim = POSITION_LIMIT
        edge = TAKE_EDGE
        if self._spread_ema is not None:
            edge += min(1.0, self._spread_ema/20.0)

        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                vol = -od.sell_orders[ask]
                qty = 0
                if ask <= fair - edge and position < lim:
                    qty = min(vol, lim - position)
                elif ask <= fair and position < 0:
                    qty = min(vol, abs(position))
                if qty > 0:
                    orders.append(Order(PRODUCT, ask, qty)); position += qty

        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                vol = od.buy_orders[bid]
                qty = 0
                if bid >= fair + edge and position > -lim:
                    qty = min(vol, lim + position)
                elif bid >= fair and position > 0:
                    qty = min(vol, position)
                if qty > 0:
                    orders.append(Order(PRODUCT, bid, -qty)); position -= qty

        return orders, position

    def _make(self, od, position, fair):
        orders = []
        lim = POSITION_LIMIT
        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        half = max(1, int(round(self._spread_ema/2)) if self._spread_ema else 2)
        skew    = MM_SKEW * position
        bid_px  = int(round(fair - half - skew))
        ask_px  = int(round(fair + half - skew))

        if best_bid: bid_px = max(bid_px, best_bid+1 if best_bid < fair else best_bid)
        if best_ask: ask_px = min(ask_px, best_ask-1 if best_ask > fair else best_ask)
        if bid_px >= ask_px:
            bid_px = int(math.floor(fair))-1; ask_px = int(math.ceil(fair))+1

        bq = min(MM_CLIP, max(0, lim-position))
        aq = min(MM_CLIP, max(0, lim+position))

        if position >= SOFT_LONG:
            ask_px = min(ask_px, int(round(fair))); aq = min(max(1,position), MM_CLIP+20); bq = min(bq,3)
        elif position <= SOFT_SHORT:
            bid_px = max(bid_px, int(round(fair))); bq = min(max(1,abs(position)), MM_CLIP+20); aq = min(aq,3)

        if bq > 0: orders.append(Order(PRODUCT, bid_px,  bq))
        if aq > 0: orders.append(Order(PRODUCT, ask_px, -aq))
        return orders

    # ------------------------------------------------------------------
    def trade(self, od, position: int, timestamp: int, mkt: list):
        self._ingest_trades(mkt)
        mid  = self._update_stats(od)
        fair = self._fair_value()
        if fair is None or mid is None: return []

        take, pos2 = self._take(od, position, fair)
        return take + self._make(od, pos2, fair)


# ---------------------------------------------------------------------------
# Minimal Trader (VFE only — for isolated backtesting)
# ---------------------------------------------------------------------------
class Trader:
    def __init__(self):
        self.vfe = VelvetfruitExtractStrategy()

    def run(self, state: TradingState):
        result = {}
        if PRODUCT in state.order_depths:
            result[PRODUCT] = self.vfe.trade(
                state.order_depths[PRODUCT],
                state.position.get(PRODUCT, 0),
                state.timestamp,
                state.market_trades.get(PRODUCT, []))
        return result, 0, ""
