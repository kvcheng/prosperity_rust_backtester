
from collections import deque
from datamodel import Order, TradingState


# ---------------------------------------------------------------------------
# Known counterparty IDs harvested from Round-4 historical trade data.
# "MARK" (commonly "Mark") is typically a noise-trader / market-taker who
# crosses the spread and telegraphs short-term directional pressure.
# Observing whether Mark is net-buying or net-selling over a rolling window
# gives a cheap, low-latency signal that we can layer on top of the standard
# market-making fair-value estimate.
# ---------------------------------------------------------------------------
MARK_ID = "Mark"          # Adjust to exact ID seen in the data if different


class HydrogelPackStrategy:
    """
    Market-making strategy for HYDROGEL_PACK (position limit 200).

    Core design
    -----------
    1. Fair-value estimation  – EMA of mid-price blended with a short-horizon
       linear-regression trend (same pattern proven in IntarianPepperRoot).
    2. Counterparty signal    – tracks Mark's net signed volume over a rolling
       window.  A strong buy signal from Mark shifts the fair-value estimate
       upward; a strong sell signal shifts it downward.
    3. Take step              – lift cheap asks / hit expensive bids when
       price is sufficiently mis-valued relative to our fair-value estimate.
    4. Make step              – post resting bid/ask around fair value with
       inventory skew to stay flat on average.
    """

    POSITION_LIMIT   = 200
    DEFAULT_START_FAIR = 2000.0   # placeholder; overwritten on first tick
    DEFAULT_HALF_SPREAD = 4
    INVENTORY_SKEW   = 0.08       # price offset per unit of inventory
    MAX_POST_SIZE    = 30         # max qty per resting order
    SOFT_LONG        = 120
    SOFT_SHORT       = -80

    # Counterparty signal parameters
    MARK_WINDOW      = 20         # how many trades to remember
    MARK_SIGNAL_SCALE = 0.4       # how many price units per 100 net Mark volume

    def __init__(self):
        self.mid_history  = deque(maxlen=40)
        self.time_history = deque(maxlen=40)
        self.ema_mid      = None
        self.move_ema     = 1.0
        self.spread_ema   = None
        self.last_mid     = None

        # Ring-buffer of (signed_qty) seen from Mark
        self.mark_trades  = deque(maxlen=self.MARK_WINDOW)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _observe_mid(self, od):
        bids, asks = od.buy_orders, od.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2
        if bids:
            return float(max(bids))
        if asks:
            return float(min(asks))
        return None

    def _update_book_stats(self, od, timestamp):
        mid = self._observe_mid(od)
        if od.buy_orders and od.sell_orders:
            spread = min(od.sell_orders) - max(od.buy_orders)
            self.spread_ema = (
                float(spread) if self.spread_ema is None
                else 0.85 * self.spread_ema + 0.15 * spread
            )
        if mid is not None:
            self.time_history.append(timestamp / 100.0)
            self.mid_history.append(mid)
            if self.last_mid is not None:
                move = abs(mid - self.last_mid)
                self.move_ema = 0.85 * self.move_ema + 0.15 * move
            self.last_mid = mid
            if self.ema_mid is None:
                self.ema_mid = mid
        return mid

    def _dynamic_alpha(self):
        spread_term = 0.0 if self.spread_ema is None else min(0.10, self.spread_ema / 100.0)
        move_term   = min(0.12, self.move_ema / 50.0)
        return max(0.05, min(0.30, 0.05 + spread_term + move_term))

    def _trend_slope(self):
        n = len(self.mid_history)
        if n < 6:
            return 0.0
        xs = list(self.time_history)
        ys = list(self.mid_history)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def _mark_signal(self):
        """
        Returns a signed net volume from Mark over the recent window.
        Positive → Mark has been net buying (price likely to rise).
        Negative → Mark has been net selling.
        """
        return sum(self.mark_trades)

    def _ingest_market_trades(self, market_trades):
        """
        Scan the list of recent trades for Mark's activity and record
        signed quantity (positive = Mark bought, negative = Mark sold).
        """
        for trade in market_trades:
            if trade.buyer == MARK_ID:
                self.mark_trades.append(trade.quantity)
            elif trade.seller == MARK_ID:
                self.mark_trades.append(-trade.quantity)

    # ------------------------------------------------------------------
    # Fair value
    # ------------------------------------------------------------------
    def fair_value(self, od, timestamp):
        mid = self._update_book_stats(od, timestamp)

        if self.ema_mid is None:
            self.ema_mid = mid if mid is not None else self.DEFAULT_START_FAIR

        if mid is not None:
            alpha = self._dynamic_alpha()
            self.ema_mid = (1 - alpha) * self.ema_mid + alpha * mid

        trend      = self._trend_slope()
        trend_fair = self.ema_mid + trend * 3.0
        base_fair  = (0.7 * self.ema_mid + 0.3 * trend_fair) if mid is not None else trend_fair

        # Counterparty adjustment
        net_mark_vol = self._mark_signal()
        cp_adjustment = (net_mark_vol / 100.0) * self.MARK_SIGNAL_SCALE

        return base_fair + cp_adjustment

    # ------------------------------------------------------------------
    # Take (aggress mis-priced resting orders)
    # ------------------------------------------------------------------
    def take(self, od, position, timestamp):
        orders = []
        fair   = self.fair_value(od, timestamp)
        limit  = self.POSITION_LIMIT

        edge = 0.5
        if self.spread_ema is not None:
            edge += min(1.5, self.spread_ema / 20.0)
        edge += min(1.0, self.move_ema / 10.0)

        buy_edge  = edge + (0.5 if position > 60 else 0.0)
        sell_edge = edge + (0.5 if position < 0 else 0.0)

        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                vol = -od.sell_orders[ask]
                qty = 0
                if ask <= fair - buy_edge and position < limit:
                    qty = min(vol, limit - position)
                elif ask <= fair and position < 0:
                    qty = min(vol, abs(position))
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", ask, qty))
                    position += qty

        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                vol = od.buy_orders[bid]
                qty = 0
                if bid >= fair + sell_edge and position > -limit:
                    qty = min(vol, limit + position)
                elif bid >= fair and position > 0:
                    qty = min(vol, position)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", bid, -qty))
                    position -= qty

        return orders

    # ------------------------------------------------------------------
    # Make (post resting quotes)
    # ------------------------------------------------------------------
    def make(self, od, position, timestamp):
        orders = []
        fair   = self.fair_value(od, timestamp)
        limit  = self.POSITION_LIMIT

        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half_spread = max(1, (best_ask - best_bid) // 2)
        elif self.spread_ema is not None:
            half_spread = max(1, int(round(self.spread_ema / 2)))
        else:
            half_spread = self.DEFAULT_HALF_SPREAD

        skew      = self.INVENTORY_SKEW * position
        bid_price = int(round(fair - half_spread - skew))
        ask_price = int(round(fair + half_spread - skew))

        # Tighten quotes toward best prices if available
        if best_bid is not None and best_bid < fair:
            bid_price = max(bid_price, best_bid + 1)
        elif best_bid is not None:
            bid_price = max(bid_price, best_bid)
        if best_ask is not None and best_ask > fair:
            ask_price = min(ask_price, best_ask - 1)
        elif best_ask is not None:
            ask_price = min(ask_price, best_ask)

        if bid_price >= ask_price:
            bid_price = int(round(fair - 1))
            ask_price = int(round(fair + 1))

        buy_qty  = min(self.MAX_POST_SIZE, max(0, limit - position))
        sell_qty = min(self.MAX_POST_SIZE, max(0, limit + position))

        # Soft inventory limits: lean aggressively to flatten
        if position >= self.SOFT_LONG:
            ask_price = min(ask_price, int(round(fair)))
            sell_qty  = min(max(1, position), self.MAX_POST_SIZE + 20)
            buy_qty   = min(buy_qty, 4)
        elif position <= self.SOFT_SHORT:
            bid_price = max(bid_price, int(round(fair)))
            buy_qty   = min(max(1, abs(position)), self.MAX_POST_SIZE + 20)
            sell_qty  = min(sell_qty, 4)

        if buy_qty > 0:
            orders.append(Order("HYDROGEL_PACK", bid_price,  buy_qty))
        if sell_qty > 0:
            orders.append(Order("HYDROGEL_PACK", ask_price, -sell_qty))

        return orders

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def trade(self, od, position, timestamp, market_trades):
        self._ingest_market_trades(market_trades)
        take_orders = self.take(od, position, timestamp)
        updated_pos = position + sum(o.quantity for o in take_orders)
        make_orders = self.make(od, updated_pos, timestamp)
        return take_orders + make_orders


# ---------------------------------------------------------------------------
# Trader class (to be extended with VELVETFRUIT strategies in later steps)
# ---------------------------------------------------------------------------
class Trader:
    def __init__(self):
        self.hydrogel = HydrogelPackStrategy()

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            position     = state.position.get(product, 0)
            mkt_trades   = state.market_trades.get(product, [])

            if product == "HYDROGEL_PACK":
                result[product] = self.hydrogel.trade(
                    order_depth, position, state.timestamp, mkt_trades
                )

        trader_data = ""
        conversions = 0
        return result, conversions, trader_data
