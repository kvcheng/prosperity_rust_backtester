from collections import deque
from datamodel import Order, TradingState


class AshCoatedOsmiumStrategy:
    POSITION_LIMIT = 80
    DEFAULT_START_FAIR = 10000.0

    def __init__(self):
        self.fair_anchor = None

    def observe_mid(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        if bids and asks:
            return (max(bids) + min(asks)) / 2
        if bids:
            return float(max(bids))
        if asks:
            return float(min(asks))
        return None

    def fair_value(self, order_depth):
        mid = self.observe_mid(order_depth)

        if self.fair_anchor is None:
            self.fair_anchor = mid if mid is not None else self.DEFAULT_START_FAIR
            return self.fair_anchor

        if mid is not None:
            self.fair_anchor = 0.85 * self.fair_anchor + 0.15 * mid

        return self.fair_anchor

    def take(self, order_depth, position: int):
        orders = []
        limit = self.POSITION_LIMIT
        fair = self.fair_value(order_depth)

        if order_depth.sell_orders:
            for ask in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask]
                qty = 0
                if ask <= fair - 1 and position < limit:
                    qty = min(ask_volume, limit - position)
                elif ask <= fair and position < 0:
                    qty = min(ask_volume, abs(position))
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                    position += qty

        if order_depth.buy_orders:
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid]
                qty = 0
                if bid >= fair + 1 and position > -limit:
                    qty = min(bid_volume, limit + position)
                elif bid >= fair and position > 0:
                    qty = min(bid_volume, position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    position -= qty

        return orders

    def make(self, order_depth, position: int):
        orders = []
        limit = self.POSITION_LIMIT
        fair = self.fair_value(order_depth)

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half_spread = max(1, (best_ask - best_bid) // 2)
        else:
            half_spread = 8

        skew = int(round(0.12 * position))
        bid_price = int(round(fair - half_spread - skew))
        ask_price = int(round(fair + half_spread - skew))

        if best_bid is not None:
            bid_price = max(bid_price, best_bid + 1 if best_bid < fair else best_bid)
        if best_ask is not None:
            ask_price = min(ask_price, best_ask - 1 if best_ask > fair else best_ask)

        if bid_price >= ask_price:
            bid_price = int(round(fair - 1))
            ask_price = int(round(fair + 1))

        buy_qty = max(0, limit - position)
        sell_qty = max(0, limit + position)

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", ask_price, -sell_qty))

        return orders

    def trade(self, order_depth, position: int):
        orders = []
        take_orders = self.take(order_depth, position)
        orders.extend(take_orders)

        updated_position = position + sum(order.quantity for order in take_orders)
        orders.extend(self.make(order_depth, updated_position))
        return orders


class IntarianPepperRootBuyAndHoldStrategy:
    POSITION_LIMIT = 80
    # SMA(8) for short-term signal, SMA(30) for long-term reference.
    SHORT_WINDOW = 8
    LONG_WINDOW = 30
    # Base threshold required for an SMA gap to count as a real trend.
    DOWNTREND_GAP = 1.2
    UPTREND_GAP = 1.2
    # Volatility lookback and scaling used to increase threshold in noisy markets.
    VOL_WINDOW = 12
    VOL_MULTIPLIER = 1.2
    
    COOLDOWN_STEPS = 6
    MIN_HOLD_STEPS = 12

    def __init__(self):
        # Queue to store recent mid prices for SMA and volatility calculations.
        self.mid_history = deque(maxlen=self.LONG_WINDOW)
        # To indicate current direction: 1 for uptrend, -1 for downtrend, 0 for none.
        self.active_side = 0
        # An index of when the active_side last switched to monitor cooldown and hold periods.
        self.last_switch_step = -10**9

    # Helper to extract a single mid price from the order book, preferring a true midpoint 
    # but falling back to either bid or ask if not possible to compute.
    def observe_mid(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        if bids and asks:
            return (max(bids) + min(asks)) / 2
        if bids:
            return float(max(bids))
        if asks:
            return float(min(asks))
        return None

    # Simple SMA calculation over mid price history. Returns None if there is not yet
    # enough data. 
    def sma(self, window: int):
        if len(self.mid_history) < window:
            return None
        values = list(self.mid_history)[-window:]
        return sum(values) / window

    # Helper to calculate volatility based on recent mid price changes. 
    # This is calculated by taking the average absolute change in the mid price over
    # a short window. 
    def recent_volatility(self):
        if len(self.mid_history) < 2:
            return None
        diffs = [abs(self.mid_history[i] - self.mid_history[i - 1]) for i in range(1, len(self.mid_history))]
        tail = diffs[-self.VOL_WINDOW:]
        if not tail:
            return None
        return sum(tail) / len(tail)

    # Helper that calculates the trend signal of the market using the SMA spread.
    # A positive spread suggests an upward drift, whilst a negative spread suggests a downward drift.
    def raw_signal_side(self):
        short_sma = self.sma(self.SHORT_WINDOW)
        long_sma = self.sma(self.LONG_WINDOW)
        vol = self.recent_volatility()

        if short_sma is None or long_sma is None or vol is None:
            return 0

        signal = short_sma - long_sma
        # Dynamic gap in case of noisy trend signals, requiring a larger SMA to confidently
        # confirm a trend. 
        dynamic_gap = max(self.UPTREND_GAP, self.DOWNTREND_GAP, self.VOL_MULTIPLIER * vol)

        if signal > dynamic_gap:
            return 1
        if signal < -dynamic_gap:
            return -1
        return 0

    # Helper that applies a cooldown and minimum hold period. This is used in the initial building
    # to avoid immediately puchasing into a new signal and/or whipsaws that can be present. 
    # Based on the unfiltered signal from the data, determine whether we are confident enough to
    # say whether the data is trending up, down or neutral. 
    def guarded_side(self, raw_side: int, timestamp: int):
        step = timestamp // 100

        # Neutral signal means no forced direction this tick.
        if raw_side == 0:
            return 0

        # If the active_side is neutral, we can take the trend as we don't have a position. 
        if self.active_side == 0:
            self.active_side = raw_side
            self.last_switch_step = step
            return raw_side

        if raw_side == self.active_side:
            return raw_side

        # If it has not been enough time since the last time we switch, or if we have not held
        # our current position for long enough, we ignore the new signal and maintain our current stance.
        elapsed = step - self.last_switch_step
        if elapsed < self.COOLDOWN_STEPS:
            return self.active_side
        if elapsed < self.MIN_HOLD_STEPS:
            return self.active_side

        # If pass both guards, can confidently switch to the new signal.
        self.active_side = raw_side
        self.last_switch_step = step
        return raw_side

    def trade(self, order_depth, position: int, timestamp: int):
        orders = []

        mid = self.observe_mid(order_depth)
        if mid is not None:
            self.mid_history.append(mid)

        # Check the current signal being given based on market data and then check
        # whether we can confidently say this signal is real.
        raw_side = self.raw_signal_side()
        side = self.guarded_side(raw_side, timestamp)

        # Attempt to max out holding in the direction of the signal.
        target_position = position
        if side > 0:
            target_position = self.POSITION_LIMIT
        elif side < 0:
            target_position = -self.POSITION_LIMIT
        else:
            return orders

        # Attempt to move holdings based on the target. 
        # If in a buy position, take lowest asks.
        # If in a sell position, take highest bids.
        if target_position > position:
            remaining_to_buy = target_position - position
            if not order_depth.sell_orders:
                return orders

            for ask in sorted(order_depth.sell_orders.keys()):
                if remaining_to_buy <= 0:
                    break

                ask_volume = -order_depth.sell_orders[ask]
                if ask_volume <= 0:
                    continue

                buy_qty = min(ask_volume, remaining_to_buy)
                if buy_qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, buy_qty))
                    remaining_to_buy -= buy_qty
        elif target_position < position:
            remaining_to_sell = position - target_position
            if not order_depth.buy_orders:
                return orders

            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_to_sell <= 0:
                    break

                bid_volume = order_depth.buy_orders[bid]
                if bid_volume <= 0:
                    continue

                sell_qty = min(bid_volume, remaining_to_sell)
                if sell_qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -sell_qty))
                    remaining_to_sell -= sell_qty

        return orders


class Trader:
    def __init__(self):
        self.osmium = AshCoatedOsmiumStrategy()
        self.pepper_root = IntarianPepperRootBuyAndHoldStrategy()

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.osmium.trade(order_depth, position)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.pepper_root.trade(order_depth, position, state.timestamp)

        trader_data = ""
        conversions = 0
        return result, conversions, trader_data
