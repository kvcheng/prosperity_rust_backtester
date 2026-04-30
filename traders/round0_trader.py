from datamodel import Order, TradingState
from typing import Dict, List


class EmeraldsStrategy:
    POSITION_LIMIT = 20
    EMERALDS_TRUE_PRICE = 10000

    def take(self, order_depth, position: int):
        orders = []
        limit = self.POSITION_LIMIT
        fair = self.EMERALDS_TRUE_PRICE

        if order_depth.sell_orders:
            for ask in order_depth.sell_orders.keys():
                ask_volume = -order_depth.sell_orders[ask]
                qty = 0
                if ask <= fair - 1 and position < limit:
                    qty = min(ask_volume, limit - position)
                elif ask <= fair and position < 0:
                    qty = min(ask_volume, abs(position))
                if qty > 0:
                    orders.append(Order("EMERALDS", ask, qty))
                    position += qty

        if order_depth.buy_orders:
            for bid in order_depth.buy_orders.keys():
                bid_volume = order_depth.buy_orders[bid]
                qty = 0
                if bid >= fair + 1 and position > -limit:
                    qty = min(bid_volume, limit + position)
                elif bid >= fair and position > 0:
                    qty = min(bid_volume, position)
                if qty > 0:
                    orders.append(Order("EMERALDS", bid, -qty))
                    position -= qty

        return orders

    def make(self, order_depth, position: int):
        orders = []
        limit = self.POSITION_LIMIT
        fair = self.EMERALDS_TRUE_PRICE

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        bid_price = fair - 7
        ask_price = fair + 7

        if best_bid is not None and best_bid < fair:
            overbid_price = best_bid + 1
            if overbid_price < fair:
                bid_price = max(bid_price, overbid_price)
            else:
                bid_price = max(bid_price, best_bid)

        if best_ask is not None and best_ask > fair:
            undercut_price = best_ask - 1
            if undercut_price > fair:
                ask_price = min(ask_price, undercut_price)
            else:
                ask_price = min(ask_price, best_ask)

        buy_qty = max(0, limit - position)
        sell_qty = max(0, limit + position)

        if buy_qty > 0:
            orders.append(Order("EMERALDS", bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order("EMERALDS", ask_price, -sell_qty))

        return orders

    def trade(self, order_depth, position: int):
        orders = []
        orders.extend(self.take(order_depth, position))

        updated_position = position
        for order in orders:
            updated_position += order.quantity

        orders.extend(self.make(order_depth, updated_position))
        return orders


class TomatoesStrategy:
    POSITION_LIMIT = 20

    def observe_mid(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders

        if bids and asks:
            return (max(bids) + min(asks)) / 2
        if bids:
            return float(max(bids))
        if asks:
            return float(min(asks))
        return float(4993)


    def take(self, order_depth, position: int):
        orders = []
        fair = self.observe_mid(order_depth)
        limit = self.POSITION_LIMIT

        if order_depth.sell_orders:
            for ask in order_depth.sell_orders.keys():
                ask_volume = -order_depth.sell_orders[ask]
                qty = 0

                if ask <= fair - 1 and position < limit:
                    qty = min(ask_volume, limit - position)
                elif ask <= fair and position < 0:
                    qty = min(ask_volume, abs(position))

                if qty > 0:
                    orders.append(Order("TOMATOES", ask, qty))
                    position += qty

        if order_depth.buy_orders:
            for bid in order_depth.buy_orders.keys():
                bid_volume = order_depth.buy_orders[bid]
                qty = 0

                if bid >= fair + 1 and position > -limit:
                    qty = min(bid_volume, limit + position)
                elif bid >= fair and position > 0:
                    qty = min(bid_volume, position)

                if qty > 0:
                    orders.append(Order("TOMATOES", bid, -qty))
                    position -= qty

        return orders

    def make(self, order_depth, position: int):
        orders = []
        fair = self.observe_mid(order_depth)
        limit = self.POSITION_LIMIT

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        diff = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else 0

        bid_price = int(round(fair - diff))
        ask_price = int(round(fair + diff))

        if best_bid is not None and best_bid < fair:
            overbid_price = best_bid + 1
            if overbid_price < fair:
                bid_price = max(bid_price, overbid_price)
            else:
                bid_price = max(bid_price, best_bid)

        if best_ask is not None and best_ask > fair:
            undercut_price = best_ask - 1
            if undercut_price > fair:
                ask_price = min(ask_price, undercut_price)
            else:
                ask_price = min(ask_price, best_ask)

        buy_qty = max(0, limit - position)
        sell_qty = max(0, limit + position)

        if buy_qty > 0:
            orders.append(Order("TOMATOES", bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order("TOMATOES", ask_price, -sell_qty))

        return orders

    def trade(self, order_depth, position: int):
        orders = []
        orders.extend(self.take(order_depth, position))

        updated_position = position
        for order in orders:
            updated_position += order.quantity

        orders.extend(self.make(order_depth, updated_position))
        return orders


class Trader:
    def __init__(self):
        self.emeralds = EmeraldsStrategy()
        self.tomatoes = TomatoesStrategy()

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            if product == "EMERALDS":
                position = state.position.get(product, 0)
                result[product] = self.emeralds.trade(order_depth, position)

            elif product == "TOMATOES":
                position = state.position.get(product, 0)
                result[product] = self.tomatoes.trade(order_depth, position)

        trader_data = ""
        conversions = 0
        return result, conversions, trader_data
