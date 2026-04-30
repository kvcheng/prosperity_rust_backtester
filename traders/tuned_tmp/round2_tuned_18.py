from collections import deque
from datamodel import Order, TradingState


class AshCoatedOsmiumStrategy:
    POSITION_LIMIT = 80
    DEFAULT_START_FAIR = 10000.0
    WINDOW = 30
    SHORT_WINDOW = 6
    LONG_WINDOW = 24
    FAST_ALPHA = 0.22
    SLOW_ALPHA = 0.08
    ENTRY_BAND = 1.0
    MAX_TAKE_PER_LEVEL = 18
    MAX_POST_SIZE = 14
    MAX_TURNOVER_PER_TICK = 32
    Z_ENTRY = 1.1
    Z_EXIT = 0.45
    MIN_TARGET_SIZE = 15
    STRONG_BEAR_TARGET = -40
    MILD_BEAR_TARGET = -10
    STRONG_BEAR_COOLDOWN = 8

    def __init__(self):
        # Round 2 Osmium behaves like a stable mean-reverting product around the 10k area.
        # We therefore keep a rolling mean and a volatility estimate, then fade extremes.
        self.mid_history = deque(maxlen=self.WINDOW)
        self.ema_mid = None
        self.vol_ema = 1.5
        self.fair_anchor = None
        self.last_fair = None
        self.fair_velocity_ema = 0.0
        self.buy_cooldown = 0

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

    def _microprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = max(1, order_depth.buy_orders[best_bid])
        ask_vol = max(1, -order_depth.sell_orders[best_ask])
        return (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)

    def _update_state(self, order_depth):
        mid = self.observe_mid(order_depth)
        micro = self._microprice(order_depth)

        if mid is not None:
            self.mid_history.append(mid)
            if self.ema_mid is None:
                self.ema_mid = mid
            else:
                self.ema_mid = (1 - self.FAST_ALPHA) * self.ema_mid + self.FAST_ALPHA * mid

            if len(self.mid_history) >= 2:
                prev = self.mid_history[-2]
                move = abs(mid - prev)
                self.vol_ema = (1 - self.SLOW_ALPHA) * self.vol_ema + self.SLOW_ALPHA * move

        if self.fair_anchor is None:
            self.fair_anchor = mid if mid is not None else self.DEFAULT_START_FAIR

        # Blend a slowly adapting anchor with the latest microstructure.
        fair = self.ema_mid if self.ema_mid is not None else self.fair_anchor
        if micro is not None:
            fair = 0.7 * fair + 0.3 * micro

        if self.mid_history:
            rolling_mean = sum(self.mid_history) / len(self.mid_history)
            fair = 0.55 * fair + 0.45 * rolling_mean

        if self.last_fair is not None:
            drift = fair - self.last_fair
            self.fair_velocity_ema = 0.75 * self.fair_velocity_ema + 0.25 * drift
        self.last_fair = fair

        self.fair_anchor = fair
        return fair, mid

    def _vol_band(self):
        # Keep the entry threshold above one tick, but widen it when noise increases.
        return max(self.ENTRY_BAND, 0.75 * self.vol_ema + 0.5)

    def _z_score(self, mid: float):
        if mid is None or len(self.mid_history) < 6:
            return 0.0

        mean = sum(self.mid_history) / len(self.mid_history)
        variance = sum((x - mean) ** 2 for x in self.mid_history) / len(self.mid_history)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (mid - mean) / std

    def _trend_regime(self):
        if len(self.mid_history) < self.LONG_WINDOW:
            return 0

        values = list(self.mid_history)
        short_ma = sum(values[-self.SHORT_WINDOW:]) / self.SHORT_WINDOW
        long_ma = sum(values[-self.LONG_WINDOW:]) / self.LONG_WINDOW
        momentum = values[-1] - values[-self.SHORT_WINDOW]

        # -2 strong downtrend, -1 mild downtrend, 0 neutral, +1 uptrend
        if short_ma <= long_ma - 1.0 and momentum <= -2.0 and self.fair_velocity_ema <= -0.35:
            return -2
        if short_ma <= long_ma - 0.5 and momentum <= -0.75:
            return -1
        if short_ma >= long_ma + 0.5 and momentum >= 0.75 and self.fair_velocity_ema >= 0.15:
            return 1
        return 0

    def _target_position(self, position: int, z_score: float, regime: int):
        # Scaled mean-reversion sizing instead of immediate full-limit flips.
        abs_z = abs(z_score)
        if abs_z < self.Z_EXIT:
            target = 0
        else:
            intensity = min(1.0, (abs_z - self.Z_EXIT) / 2.15)
            target_size = int(round(self.MIN_TARGET_SIZE + intensity * (self.POSITION_LIMIT - self.MIN_TARGET_SIZE)))
            target = target_size if z_score < 0 else -target_size

        # Fast bearish override for the drawdown regime seen in day 1:
        # stop catching falling knives and unwind/flip faster.
        if regime == -2:
            target = min(target, self.STRONG_BEAR_TARGET)
        elif regime == -1:
            target = min(target, self.MILD_BEAR_TARGET)

        if self.buy_cooldown > 0 and target > 0:
            target = 0

        return max(-self.POSITION_LIMIT, min(self.POSITION_LIMIT, target))

    def take(self, order_depth, position: int, fair: float, z: float, regime: int, target_position: int):
        orders = []
        limit = self.POSITION_LIMIT
        band = self._vol_band()
        remaining_turnover = self.MAX_TURNOVER_PER_TICK

        # Stop-loss style de-risking for stale longs during downward drift.
        if position > 0 and regime <= -1 and self.fair_velocity_ema <= -0.25 and order_depth.buy_orders:
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_turnover <= 0 or position <= 0:
                    break
                bid_volume = order_depth.buy_orders[bid]
                qty = min(bid_volume, position, remaining_turnover, self.MAX_TAKE_PER_LEVEL)
                if qty > 0 and bid >= fair - 2:
                    orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    position -= qty
                    remaining_turnover -= qty

        # Prioritize moving current inventory toward target position quickly.
        if target_position < position and order_depth.buy_orders and remaining_turnover > 0:
            remaining_to_sell = position - target_position
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_to_sell <= 0 or remaining_turnover <= 0:
                    break
                bid_volume = order_depth.buy_orders[bid]
                qty = min(bid_volume, remaining_to_sell, self.MAX_TAKE_PER_LEVEL, remaining_turnover)
                if qty > 0 and bid >= fair - 1:
                    orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    position -= qty
                    remaining_to_sell -= qty
                    remaining_turnover -= qty

        if target_position > position and order_depth.sell_orders and remaining_turnover > 0:
            remaining_to_buy = target_position - position
            for ask in sorted(order_depth.sell_orders.keys()):
                if remaining_to_buy <= 0 or remaining_turnover <= 0:
                    break
                ask_volume = -order_depth.sell_orders[ask]
                qty = min(ask_volume, remaining_to_buy, self.MAX_TAKE_PER_LEVEL, remaining_turnover)
                if qty > 0 and ask <= fair + 1:
                    orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                    position += qty
                    remaining_to_buy -= qty
                    remaining_turnover -= qty

        # Additional opportunistic crossing when price is far from fair.
        if order_depth.sell_orders and remaining_turnover > 0:
            for ask in sorted(order_depth.sell_orders.keys()):
                if remaining_turnover <= 0:
                    break
                ask_volume = -order_depth.sell_orders[ask]
                qty = 0

                if ask <= fair - band and position < limit and regime >= 0 and self.buy_cooldown == 0:
                    qty = min(ask_volume, limit - position, self.MAX_TAKE_PER_LEVEL, remaining_turnover)
                elif z < -self.Z_ENTRY and ask <= fair and position < limit and regime >= 0 and self.buy_cooldown == 0:
                    qty = min(ask_volume, limit - position, self.MAX_TAKE_PER_LEVEL, remaining_turnover)

                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                    position += qty
                    remaining_turnover -= qty

        if order_depth.buy_orders and remaining_turnover > 0:
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_turnover <= 0:
                    break
                bid_volume = order_depth.buy_orders[bid]
                qty = 0

                if bid >= fair + band and position > -limit:
                    qty = min(bid_volume, limit + position, self.MAX_TAKE_PER_LEVEL, remaining_turnover)
                elif z > self.Z_ENTRY and bid >= fair and position > -limit:
                    qty = min(bid_volume, limit + position, self.MAX_TAKE_PER_LEVEL, remaining_turnover)
                elif regime <= -1 and bid >= fair - 1 and position > -limit:
                    qty = min(bid_volume, limit + position, self.MAX_TAKE_PER_LEVEL, remaining_turnover)

                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    position -= qty
                    remaining_turnover -= qty

        return orders

    def make(self, order_depth, position: int, fair: float, z: float, regime: int, target_position: int):
        orders = []
        limit = self.POSITION_LIMIT
        band = self._vol_band()

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half_spread = max(1, (best_ask - best_bid) // 2)
        else:
            half_spread = max(2, int(round(band)))

        # Inventory and z-score both skew the quotes back toward the mean.
        skew = int(round(0.10 * position + 0.8 * z))
        bid_price = int(round(fair - half_spread - skew))
        ask_price = int(round(fair + half_spread - skew))

        if best_bid is not None:
            bid_price = max(bid_price, best_bid + 1 if best_bid < fair else best_bid)
        if best_ask is not None:
            ask_price = min(ask_price, best_ask - 1 if best_ask > fair else best_ask)

        if bid_price >= ask_price:
            bid_price = int(round(fair - 1))
            ask_price = int(round(fair + 1))

        buy_qty = min(self.MAX_POST_SIZE, max(0, limit - position))
        sell_qty = min(self.MAX_POST_SIZE, max(0, limit + position))

        # Regime-aware quoting: in downtrends, de-risk bids and lean into asks.
        if regime == -2:
            buy_qty = min(buy_qty, 2)
            sell_qty = min(limit + position, self.MAX_POST_SIZE + 8)
            bid_price = min(bid_price, int(round(fair - 2)))
            ask_price = min(ask_price, int(round(fair)))
        elif regime == -1:
            buy_qty = min(buy_qty, 4)
            sell_qty = min(limit + position, self.MAX_POST_SIZE + 6)
            bid_price = min(bid_price, int(round(fair - 1)))
            ask_price = min(ask_price, int(round(fair + 1)))

        # If we are above fair, bias into sells; below fair, bias into buys.
        if z > 0.75:
            buy_qty = min(buy_qty, 4)
            sell_qty = min(limit + position, self.MAX_POST_SIZE + 4)
        elif z < -0.75:
            buy_qty = min(limit - position, self.MAX_POST_SIZE + 4)
            sell_qty = min(sell_qty, 4)

        if position >= 40:
            ask_price = min(ask_price, int(round(fair)))
            sell_qty = min(max(1, position), self.MAX_POST_SIZE + 6)
        elif position <= -40:
            bid_price = max(bid_price, int(round(fair)))
            buy_qty = min(max(1, abs(position)), self.MAX_POST_SIZE + 6)

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", ask_price, -sell_qty))

        return orders

    def trade(self, order_depth, position: int):
        fair, mid = self._update_state(order_depth)
        z = self._z_score(mid)
        regime = self._trend_regime()

        if regime == -2:
            self.buy_cooldown = self.STRONG_BEAR_COOLDOWN
        elif self.buy_cooldown > 0:
            self.buy_cooldown -= 1

        target_position = self._target_position(position, z, regime)

        orders = []
        take_orders = self.take(order_depth, position, fair, z, regime, target_position)
        orders.extend(take_orders)

        updated_position = position + sum(order.quantity for order in take_orders)
        orders.extend(self.make(order_depth, updated_position, fair, z, regime, target_position))
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

    def bid(self):
        return 28

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
