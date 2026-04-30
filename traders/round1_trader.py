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
        bid_price = fair - half_spread - skew
        ask_price = fair + half_spread - skew

        if best_bid is not None:
            bid_price = max(bid_price, best_bid + 1 if best_bid < fair else best_bid)
        if best_ask is not None:
            ask_price = min(ask_price, best_ask - 1 if best_ask > fair else best_ask)

        if bid_price >= ask_price:
            bid_price = fair - 1
            ask_price = fair + 1

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


class IntarianPepperRootStrategy:
    POSITION_LIMIT = 80
    DEFAULT_START_FAIR = 11000.0
    DEFAULT_HALF_SPREAD = 6
    INVENTORY_SKEW = 0.12
    MAX_POST_SIZE = 12
    SOFT_LONG = 50
    SOFT_SHORT = -30

    def __init__(self):
        # In case the actual data trends downwards instead of upwards, I decided to 
        # add a linear regression to estimate the trend. This is done with 
        # rolling observations. The mid price history is used to estimate the local trend,
        # and the time history is used to compute the slope.
        self.mid_history = deque(maxlen=30)
        self.time_history = deque(maxlen=30)
        self.ema_mid = None
        self.move_ema = 1.0
        self.spread_ema = None
        self.last_mid = None
        self.last_timestamp = None

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

    def _update_observations(self, order_depth, timestamp: int):
        mid = self.observe_mid(order_depth)

        if order_depth.buy_orders and order_depth.sell_orders:
            spread = min(order_depth.sell_orders) - max(order_depth.buy_orders)
            if self.spread_ema is None:
                self.spread_ema = float(spread)
            else:
                self.spread_ema = 0.85 * self.spread_ema + 0.15 * spread

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
        # Higher spread / movement means we should respond faster to fresh data.
        spread_term = 0.0 if self.spread_ema is None else min(0.10, self.spread_ema / 100.0)
        move_term = min(0.12, self.move_ema / 50.0)
        alpha = 0.05 + spread_term + move_term
        return max(0.05, min(0.30, alpha))

    def _trend_slope(self):
        # Simple linear regression slope over the recent window.
        # Positive slope forecasts an upward drift; negative slope forecasts a downward drift.
        n = len(self.mid_history)
        if n < 6:
            return 0.0

        xs = list(self.time_history)
        ys = list(self.mid_history)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        return numer / denom

    def fair_value(self, order_depth, timestamp: int):
        mid = self._update_observations(order_depth, timestamp)

        if self.ema_mid is None:
            self.ema_mid = mid if mid is not None else self.DEFAULT_START_FAIR

        if mid is not None:
            alpha = self._dynamic_alpha()
            self.ema_mid = (1 - alpha) * self.ema_mid + alpha * mid

        trend = self._trend_slope()

        # Predict a few book updates into the future using the local trend estimate.
        # This replaces the old fixed upward drift assumption.
        horizon_steps = 3.0
        trend_fair = self.ema_mid + trend * horizon_steps

        if mid is None:
            return trend_fair

        # Blend the smoothed level with the regression forecast.
        return 0.7 * self.ema_mid + 0.3 * trend_fair

    def take(self, order_depth, position: int, timestamp: int):
        orders = []
        fair = self.fair_value(order_depth, timestamp)
        limit = self.POSITION_LIMIT

        # Thresholds adapt to the observed spread and movement, so the strategy
        # becomes more cautious when the market is noisy and more aggressive when calm.
        edge = 0.5
        if self.spread_ema is not None:
            edge += min(1.5, self.spread_ema / 20.0)
        edge += min(1.0, self.move_ema / 10.0)

        buy_edge = edge
        sell_edge = edge
        if position > 40:
            buy_edge = edge + 0.5
            sell_edge = max(0.25, edge - 0.25)
        elif position < 0:
            buy_edge = max(0.25, edge - 0.25)
            sell_edge = edge + 0.5

        if order_depth.sell_orders:
            for ask in sorted(order_depth.sell_orders.keys()):
                ask_volume = -order_depth.sell_orders[ask]
                qty = 0
                if ask <= fair - buy_edge and position < limit:
                    qty = min(ask_volume, limit - position)
                elif ask <= fair and position < 0:
                    qty = min(ask_volume, abs(position))
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                    position += qty

        if order_depth.buy_orders:
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                bid_volume = order_depth.buy_orders[bid]
                qty = 0
                if bid >= fair + sell_edge and position > -limit:
                    qty = min(bid_volume, limit + position)
                elif bid >= fair and position > 0:
                    qty = min(bid_volume, position)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                    position -= qty

        return orders

    def make(self, order_depth, position: int, timestamp: int):
        orders = []
        fair = self.fair_value(order_depth, timestamp)
        limit = self.POSITION_LIMIT

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half_spread = max(1, (best_ask - best_bid) // 2)
        else:
            half_spread = self.DEFAULT_HALF_SPREAD
            if self.spread_ema is not None:
                half_spread = max(1, int(round(self.spread_ema / 2)))

        skew = self.INVENTORY_SKEW * position
        bid_price = int(round(fair - half_spread - skew))
        ask_price = int(round(fair + half_spread - skew))

        if position <= 0:
            bid_price += 1
            ask_price += 2
        elif position > 20:
            bid_price -= 1

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

        buy_qty = min(self.MAX_POST_SIZE, max(0, limit - position))
        sell_qty = min(self.MAX_POST_SIZE, max(0, limit + position))

        if position >= self.SOFT_LONG:
            ask_price = min(ask_price, int(round(fair)))
            sell_qty = min(max(1, position), self.MAX_POST_SIZE + 8)
            buy_qty = min(buy_qty, 3)
        elif position <= self.SOFT_SHORT:
            bid_price = max(bid_price, int(round(fair)))
            buy_qty = min(max(1, abs(position)), self.MAX_POST_SIZE + 8)
            sell_qty = min(sell_qty, 4)

        if buy_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, -sell_qty))

        return orders

    def trade(self, order_depth, position: int, timestamp: int):
        orders = []
        take_orders = self.take(order_depth, position, timestamp)
        orders.extend(take_orders)

        updated_position = position + sum(order.quantity for order in take_orders)
        orders.extend(self.make(order_depth, updated_position, timestamp))
        return orders


class Trader:
    def __init__(self):
        self.osmium = AshCoatedOsmiumStrategy()
        self.pepper_root = IntarianPepperRootStrategy()

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
