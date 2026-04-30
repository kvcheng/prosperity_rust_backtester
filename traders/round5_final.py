from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


def _best_bid_ask(order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def _observe_mid(order_depth: OrderDepth) -> Optional[float]:
    best_bid, best_ask = _best_bid_ask(order_depth)
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    if best_bid is not None:
        return float(best_bid)
    if best_ask is not None:
        return float(best_ask)
    return None


def _ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return (1.0 - alpha) * prev + alpha * x


def _linreg_slope(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 6:
        return 0.0
    xs = xs[-n:]
    ys = ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    numer = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return numer / denom


class Trader:
    ROUND5_PREFIXES = (
        "GALAXY_SOUNDS_",
        "SLEEP_POD_",
        "MICROCHIP_",
        "PEBBLES_",
        "ROBOT_",
        "UV_VISOR_",
        "TRANSLATOR_",
        "PANEL_",
        "OXYGEN_SHAKE_",
        "SNACKPACK_",
    )

    LIMIT = 10

    
    TRADE_PRODUCTS = {
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
    }

    
    CHOC = "SNACKPACK_CHOCOLATE"
    VAN = "SNACKPACK_VANILLA"

    
    HIST_LEN = 20
    BASE_ALPHA = 0.12
    TREND_HORIZON = 3.0

    
    QUOTE_SIZE = 1
    INV_SKEW_TICKS = 0.5  
    PAIR_EXPOSURE_SKEW = 0.3  

    
    FLATTEN_TS = 995_000

    def run(self, state: TradingState):
        data = self._load_state(state.traderData)

        
        mids: Dict[str, float] = {}
        books: Dict[str, Tuple[Optional[int], Optional[int], int]] = {}
        

        for product, order_depth in state.order_depths.items():
            if not self._is_round5(product):
                continue
            if product not in self.TRADE_PRODUCTS:
                continue

            best_bid, best_ask = _best_bid_ask(order_depth)
            if best_bid is None or best_ask is None:
                continue
            if best_ask <= best_bid:
                continue

            mid = _observe_mid(order_depth)
            if mid is None:
                continue

            mids[product] = mid
            books[product] = (best_bid, best_ask, best_ask - best_bid)

            self._update_product_stats(data, product, state.timestamp, mid, best_ask - best_bid)

        
        if self.CHOC in mids and self.VAN in mids:
            sum_mid = mids[self.CHOC] + mids[self.VAN]
            data.setdefault("pair", {})
            pair = data["pair"]
            
            pair["choc_van_sum_ema"] = _ema(pair.get("choc_van_sum_ema"), sum_mid, 0.02)

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if not self._is_round5(product):
                orders_by_product[product] = []
                continue
            if product not in self.TRADE_PRODUCTS:
                orders_by_product[product] = []
                continue

            position = int(state.position.get(product, 0))
            best_bid, best_ask = _best_bid_ask(order_depth)
            if best_bid is None or best_ask is None or best_ask <= best_bid:
                orders_by_product[product] = []
                continue

            fair = self._fair_value(data, product)

            
            pair_exposure = 0
            if product in (self.CHOC, self.VAN):
                pair_exposure = int(state.position.get(self.CHOC, 0)) + int(state.position.get(self.VAN, 0))

            
            pair = data.get("pair", {})
            sum_anchor = pair.get("choc_van_sum_ema")
            if sum_anchor is not None:
                if product == self.CHOC and self.VAN in mids:
                    implied = float(sum_anchor) - mids[self.VAN]
                    fair = 0.40 * fair + 0.60 * implied
                elif product == self.VAN and self.CHOC in mids:
                    implied = float(sum_anchor) - mids[self.CHOC]
                    fair = 0.40 * fair + 0.60 * implied

            if state.timestamp >= self.FLATTEN_TS:
                orders_by_product[product] = self._flatten(order_depth, product, position)
                continue

            orders: List[Order] = []

            
            orders.extend(self._take(order_depth, product, position, fair))
            position2 = position + sum(o.quantity for o in orders)

            
            effective_position = position2
            if product in (self.CHOC, self.VAN) and pair_exposure != 0:
                effective_position = int(round(position2 + self.PAIR_EXPOSURE_SKEW * pair_exposure))

            orders.extend(self._make(order_depth, product, effective_position, fair))
            orders_by_product[product] = orders

        trader_data_out = json.dumps(data, separators=(",", ":"))
        return orders_by_product, 0, trader_data_out

    
    
    

    def _load_state(self, trader_data: str):
        if not trader_data:
            return {"p": {}, "pair": {}}
        try:
            d = json.loads(trader_data)
            if not isinstance(d, dict):
                return {"p": {}, "pair": {}}
            d.setdefault("p", {})
            d.setdefault("pair", {})
            return d
        except Exception:
            return {"p": {}, "pair": {}}

    def _update_product_stats(self, data, product: str, timestamp: int, mid: float, spread: int):
        pdata = data.setdefault("p", {}).setdefault(product, {})

        
        t = timestamp / 100.0

        ts_hist = pdata.get("ts") or []
        mid_hist = pdata.get("mid") or []
        ts_hist.append(t)
        mid_hist.append(mid)
        if len(ts_hist) > self.HIST_LEN:
            ts_hist = ts_hist[-self.HIST_LEN :]
            mid_hist = mid_hist[-self.HIST_LEN :]
        pdata["ts"] = ts_hist
        pdata["mid"] = mid_hist

        last_mid = pdata.get("last_mid")
        if last_mid is not None:
            move = abs(mid - float(last_mid))
            pdata["move_ema"] = _ema(pdata.get("move_ema"), move, 0.15)
        pdata["last_mid"] = mid

        pdata["spread_ema"] = _ema(pdata.get("spread_ema"), float(spread), 0.10)

        
        move_ema = float(pdata.get("move_ema") or 1.0)
        spread_ema = float(pdata.get("spread_ema") or spread)
        alpha = self.BASE_ALPHA + min(0.10, spread_ema / 200.0) + min(0.10, move_ema / 100.0)
        alpha = max(0.05, min(0.30, alpha))

        pdata["ema_mid"] = _ema(pdata.get("ema_mid"), mid, alpha)
        pdata["alpha"] = alpha

        
        pdata["slope"] = _linreg_slope(ts_hist, mid_hist)

    def _fair_value(self, data, product: str) -> float:
        pdata = data.get("p", {}).get(product, {})
        ema_mid = pdata.get("ema_mid")
        if ema_mid is None:
            return 10_000.0
        slope = float(pdata.get("slope") or 0.0)
        
        return float(ema_mid) + slope * self.TREND_HORIZON

    
    
    

    def _flatten(self, od: OrderDepth, product: str, position: int) -> List[Order]:
        best_bid, best_ask = _best_bid_ask(od)
        if position == 0 or best_bid is None or best_ask is None:
            return []
        if position > 0:
            return [Order(product, best_bid, -position)]
        return [Order(product, best_ask, -position)]

    def _take(self, od: OrderDepth, product: str, position: int, fair: float) -> List[Order]:
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return []

        spread = best_ask - best_bid
        half = max(1, spread // 2)
        take_edge = max(2, half + 1)

        orders: List[Order] = []

        
        if best_ask <= fair - take_edge and position < self.LIMIT:
            ask_vol = max(0, -int(od.sell_orders.get(best_ask, 0)))
            qty = min(self.QUOTE_SIZE, ask_vol, self.LIMIT - position)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        
        if best_bid >= fair + take_edge and position > -self.LIMIT:
            bid_vol = max(0, int(od.buy_orders.get(best_bid, 0)))
            qty = min(self.QUOTE_SIZE, bid_vol, self.LIMIT + position)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty

        return orders

    def _make(self, od: OrderDepth, product: str, position: int, fair: float) -> List[Order]:
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return []

        spread = best_ask - best_bid

        
        inside_bid = best_bid + 1 if spread > 1 else best_bid
        inside_ask = best_ask - 1 if spread > 1 else best_ask

        
        skew = int(round(self.INV_SKEW_TICKS * position))

        
        
        target = int(round(fair)) - skew
        desired_bid = target - 1
        desired_ask = target + 1

        max_bid = inside_ask - 1
        min_ask = inside_bid + 1

        bid_px = max(best_bid, min(max_bid, desired_bid))
        ask_px = min(best_ask, max(min_ask, desired_ask))

        
        if bid_px >= ask_px:
            bid_px, ask_px = inside_bid, inside_ask
        if bid_px >= ask_px:
            bid_px, ask_px = best_bid, best_ask

        buy_qty = min(self.QUOTE_SIZE, max(0, self.LIMIT - position))
        sell_qty = min(self.QUOTE_SIZE, max(0, self.LIMIT + position))

        orders: List[Order] = []
        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))
        return orders

    def _is_round5(self, product: str) -> bool:
        return product.startswith(self.ROUND5_PREFIXES)
