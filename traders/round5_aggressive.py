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

    SNACKS = [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        # "SNACKPACK_PISTACHIO",
        # "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ]
    CHOC = "SNACKPACK_CHOCOLATE"
    VAN = "SNACKPACK_VANILLA"

    EXTRA_PRODUCTS = [
        # "PEBBLES_XL",
    ]

    TRADE_PRODUCTS = set(SNACKS + EXTRA_PRODUCTS)

    # State/tuning
    HIST_LEN = 30
    BASE_ALPHA = 0.10
    TREND_HORIZON = 2.5

    # Aggressiveness
    QUOTE_SIZE = 2
    INV_SKEW_TICKS = 0.45
    GROUP_EXPOSURE_SKEW = 0.18  # skew based on net snack exposure

    # Per-product caution list based on bundled round5 behaviour.
    LOW_CONF_SNACKS = {
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
    }
    QUOTE_SIZE_BY_PRODUCT = {
        "SNACKPACK_PISTACHIO": 1,
        "SNACKPACK_STRAWBERRY": 1,
    }
    MAKER_WIDEN_BY_PRODUCT = {
        "SNACKPACK_PISTACHIO": 2,
        "SNACKPACK_STRAWBERRY": 2,
    }

    PEBBLES_PRODUCT = "PEBBLES_XL"
    PEBBLES_QUOTE_SIZE = 1
    PEBBLES_INV_SKEW_TICKS = 0.20

    # Cross-sectional mean reversion strength for snack cluster fair values
    SNACK_INDEX_WEIGHT = 0.35

    # Complement anchor smoothing and weight
    SUM_ANCHOR_ALPHA = 0.03
    CHOC_VAN_IMPLIED_WEIGHT = 0.55

    # End-of-day risk reduction
    FLATTEN_TS = 995_000

    def run(self, state: TradingState):
        data = self._load_state(state.traderData)

        mids: Dict[str, float] = {}

        # Update per-product stats.
        for product, order_depth in state.order_depths.items():
            if not self._is_round5(product) or product not in self.TRADE_PRODUCTS:
                continue
            best_bid, best_ask = _best_bid_ask(order_depth)
            if best_bid is None or best_ask is None or best_ask <= best_bid:
                continue
            mid = _observe_mid(order_depth)
            if mid is None:
                continue
            mids[product] = mid
            self._update_product_stats(data, product, state.timestamp, mid, best_ask - best_bid)

        # Update CHOC/VAN complement sum anchor.
        if self.CHOC in mids and self.VAN in mids:
            pair = data.setdefault("pair", {})
            sum_mid = mids[self.CHOC] + mids[self.VAN]
            pair["choc_van_sum_ema"] = _ema(pair.get("choc_van_sum_ema"), sum_mid, self.SUM_ANCHOR_ALPHA)

        # Precompute snack "index" fair: mean of available snack EMA mids.
        snack_emas: List[float] = []
        for p in self.SNACKS:
            v = data.get("p", {}).get(p, {}).get("ema_mid")
            if v is not None:
                snack_emas.append(float(v))
        snack_index = (sum(snack_emas) / len(snack_emas)) if snack_emas else None

        snack_exposure = sum(int(state.position.get(p, 0)) for p in self.SNACKS)

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

            if state.timestamp >= self.FLATTEN_TS:
                orders_by_product[product] = self._flatten(order_depth, product, position)
                continue

            fair = self._fair_value(data, product)

            # Snack index anchoring: pull each snack's fair towards the cluster index.
            if product in self.SNACKS and snack_index is not None:
                fair = (1.0 - self.SNACK_INDEX_WEIGHT) * fair + self.SNACK_INDEX_WEIGHT * snack_index

            # CHOC/VAN complement implied fair.
            sum_anchor = data.get("pair", {}).get("choc_van_sum_ema")
            if sum_anchor is not None:
                if product == self.CHOC and self.VAN in mids:
                    implied = float(sum_anchor) - mids[self.VAN]
                    fair = (1.0 - self.CHOC_VAN_IMPLIED_WEIGHT) * fair + self.CHOC_VAN_IMPLIED_WEIGHT * implied
                elif product == self.VAN and self.CHOC in mids:
                    implied = float(sum_anchor) - mids[self.CHOC]
                    fair = (1.0 - self.CHOC_VAN_IMPLIED_WEIGHT) * fair + self.CHOC_VAN_IMPLIED_WEIGHT * implied

            orders: List[Order] = []

            # Taker: only when best price is far from fair.
            # For PEBBLES_XL, disable taker entirely (it was the main drawdown driver).
            if product != self.PEBBLES_PRODUCT and product not in self.LOW_CONF_SNACKS:
                orders.extend(self._take(order_depth, product, position, fair))
            position2 = position + sum(o.quantity for o in orders)

            # Maker: skew by own inventory; for snacks also skew by net group exposure.
            effective_position = position2
            if product in self.SNACKS and snack_exposure != 0:
                effective_position = int(round(position2 + self.GROUP_EXPOSURE_SKEW * snack_exposure))

            if product == self.PEBBLES_PRODUCT:
                orders.extend(self._make_pebbles(order_depth, product, position2, fair))
            else:
                orders.extend(self._make(order_depth, product, effective_position, fair))
            orders_by_product[product] = orders

        trader_data_out = json.dumps(data, separators=(",", ":"))
        return orders_by_product, 0, trader_data_out

    # ------------------------------------------------------------------ state
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
        alpha = self.BASE_ALPHA + min(0.10, spread_ema / 180.0) + min(0.10, move_ema / 90.0)
        alpha = max(0.05, min(0.28, alpha))
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

    # ------------------------------------------------------------------ orders
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
        take_edge = max(2, half + 2)

        quote_size = int(self.QUOTE_SIZE_BY_PRODUCT.get(product, self.QUOTE_SIZE))
        orders: List[Order] = []

        if best_ask <= fair - take_edge and position < self.LIMIT:
            ask_vol = max(0, -int(od.sell_orders.get(best_ask, 0)))
            qty = min(quote_size, ask_vol, self.LIMIT - position)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        if best_bid >= fair + take_edge and position > -self.LIMIT:
            bid_vol = max(0, int(od.buy_orders.get(best_bid, 0)))
            qty = min(quote_size, bid_vol, self.LIMIT + position)
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

        quote_size = int(self.QUOTE_SIZE_BY_PRODUCT.get(product, self.QUOTE_SIZE))
        widen = int(self.MAKER_WIDEN_BY_PRODUCT.get(product, 1))

        skew = int(round(self.INV_SKEW_TICKS * position))
        target = int(round(fair)) - skew

        desired_bid = target - widen
        desired_ask = target + widen
        max_bid = inside_ask - 1
        min_ask = inside_bid + 1

        bid_px = max(best_bid, min(max_bid, desired_bid))
        ask_px = min(best_ask, max(min_ask, desired_ask))

        if bid_px >= ask_px:
            bid_px, ask_px = inside_bid, inside_ask
        if bid_px >= ask_px:
            bid_px, ask_px = best_bid, best_ask

        buy_qty = min(quote_size, max(0, self.LIMIT - position))
        sell_qty = min(quote_size, max(0, self.LIMIT + position))

        orders: List[Order] = []
        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))
        return orders

    def _make_pebbles(self, od: OrderDepth, product: str, position: int, fair: float) -> List[Order]:
        """More conservative maker for PEBBLES_XL.

        - Smaller quote size.
        - One-sided quoting to reduce inventory when position != 0.
        """
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return []

        spread = best_ask - best_bid
        inside_bid = best_bid + 1 if spread > 1 else best_bid
        inside_ask = best_ask - 1 if spread > 1 else best_ask

        skew = int(round(self.PEBBLES_INV_SKEW_TICKS * position))
        target = int(round(fair)) - skew
        desired_bid = target - 2
        desired_ask = target + 2

        max_bid = inside_ask - 1
        min_ask = inside_bid + 1

        bid_px = max(best_bid, min(max_bid, desired_bid))
        ask_px = min(best_ask, max(min_ask, desired_ask))

        if bid_px >= ask_px:
            bid_px, ask_px = inside_bid, inside_ask
        if bid_px >= ask_px:
            bid_px, ask_px = best_bid, best_ask

        buy_qty = min(self.PEBBLES_QUOTE_SIZE, max(0, self.LIMIT - position))
        sell_qty = min(self.PEBBLES_QUOTE_SIZE, max(0, self.LIMIT + position))

        orders: List[Order] = []

        # One-sided inventory reduction.
        if position > 0:
            if sell_qty > 0:
                orders.append(Order(product, ask_px, -sell_qty))
            return orders
        if position < 0:
            if buy_qty > 0:
                orders.append(Order(product, bid_px, buy_qty))
            return orders

        # Flat: quote both sides.
        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))
        return orders

    def _is_round5(self, product: str) -> bool:
        return product.startswith(self.ROUND5_PREFIXES)
