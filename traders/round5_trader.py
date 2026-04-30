from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json, math

class Trader:
    PAIRS = [
        ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO"),
        ("SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY"),
    ]

    def __init__(self):
        self.limit = 10
        self.window = 80
        self.entry_z = 2.2
        self.exit_z = 0.5
        self.stop_z = 3.6
        self.spread_hist = {}

    def mid(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        if od.buy_orders:
            return float(max(od.buy_orders.keys()))
        if od.sell_orders:
            return float(min(od.sell_orders.keys()))
        return None

    def best_bid(self, od: OrderDepth):
        if not od.buy_orders:
            return None, 0
        px = max(od.buy_orders.keys())
        return px, od.buy_orders[px]

    def best_ask(self, od: OrderDepth):
        if not od.sell_orders:
            return None, 0
        px = min(od.sell_orders.keys())
        return px, -od.sell_orders[px]

    def buy_capacity(self, pos: int) -> int:
        return max(0, self.limit - pos)

    def sell_capacity(self, pos: int) -> int:
        return max(0, self.limit + pos)

    def append_order(self, result, product, price, qty):
        if qty == 0:
            return
        result.setdefault(product, []).append(Order(product, price, qty))

    def flatten_one(self, product, od, pos, result):
        if pos > 0:
            bid, bid_qty = self.best_bid(od)
            if bid is not None:
                qty = min(pos, bid_qty)
                self.append_order(result, product, bid, -qty)
        elif pos < 0:
            ask, ask_qty = self.best_ask(od)
            if ask is not None:
                qty = min(-pos, ask_qty)
                self.append_order(result, product, ask, qty)

    def pair_trade(self, a, b, state: TradingState, result):
        if a not in state.order_depths or b not in state.order_depths:
            return

        oda = state.order_depths[a]
        odb = state.order_depths[b]
        ma = self.mid(oda)
        mb = self.mid(odb)
        if ma is None or mb is None:
            return

        spread = ma - mb
        key = f"{a}|{b}"
        hist = self.spread_hist.get(key, [])
        hist.append(spread)
        if len(hist) > self.window:
            hist = hist[-self.window:]
        self.spread_hist[key] = hist

        if len(hist) < 40:
            return

        mu = sum(hist) / len(hist)
        var = sum((x - mu) * (x - mu) for x in hist) / len(hist)
        sd = math.sqrt(max(var, 1e-6))
        z = (spread - mu) / sd

        posa = state.position.get(a, 0)
        posb = state.position.get(b, 0)

        
        ask_a, ask_a_qty = self.best_ask(oda)
        bid_a, bid_a_qty = self.best_bid(oda)
        ask_b, ask_b_qty = self.best_ask(odb)
        bid_b, bid_b_qty = self.best_bid(odb)
        if None in (ask_a, bid_a, ask_b, bid_b):
            return

        
        if abs(z) < self.exit_z or abs(z) > self.stop_z:
            self.flatten_one(a, oda, posa, result)
            self.flatten_one(b, odb, posb, result)
            return

        
        if z > self.entry_z:
            
            qty = min(
                self.sell_capacity(posa),
                self.buy_capacity(posb),
                bid_a_qty,
                ask_b_qty,
                4,
            )
            if qty > 0:
                self.append_order(result, a, bid_a, -qty)
                self.append_order(result, b, ask_b, qty)
        elif z < -self.entry_z:
            
            qty = min(
                self.buy_capacity(posa),
                self.sell_capacity(posb),
                ask_a_qty,
                bid_b_qty,
                4,
            )
            if qty > 0:
                self.append_order(result, a, ask_a, qty)
                self.append_order(result, b, bid_b, -qty)

    def run(self, state: TradingState):
        result = {}
        for a, b in self.PAIRS:
            self.pair_trade(a, b, state, result)
        return result, 0, ""
