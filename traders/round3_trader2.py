from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json
import math


class Trader:
    LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    OPTION_SYMBOLS = [
        "VEV_4000",
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
        "VEV_6000",
        "VEV_6500",
    ]

    ACTIVE_SCALPING_SYMBOLS = {"VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}

    STRIKES = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    
    
    SMILE_A = 0.14851457
    SMILE_B = -0.00227349
    SMILE_C = 0.22683284

    
    DAYS_PER_YEAR = 365.0
    ROUND3_START_TTE_DAYS = 5.0

    
    THEO_NORM_WINDOW = 120
    IV_SCALPING_WINDOW = 60
    IV_SCALPING_ON_THRESHOLD = 0.50
    OPEN_Z = 0.95
    CLOSE_Z = 0.35
    LOW_VEGA_ADJ = 0.35
    MAX_OPTION_TAKE = 10

    
    UNDERLYING_EMA_WINDOW = 36
    UNDERLYING_THR = 10.0
    UNDERLYING_MAX_TAKE = 10
    UNDERLYING_SOFT_LIMIT = 50

    
    HYDRO_QUOTE_SIZE = 8

    def _load_data(self, state: TradingState) -> dict:
        raw = getattr(state, "traderData", "")
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _dump_data(self, data: dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))
        except Exception:
            return ""

    def _ema(self, data: dict, key: str, window: int, value: float) -> float:
        alpha = 2.0 / (window + 1.0)
        old = float(data.get(key, value))
        new = alpha * value + (1.0 - alpha) * old
        data[key] = new
        return new

    def _best_bid_ask(self, depth: OrderDepth) -> Tuple[int, int]:
        if not depth.buy_orders or not depth.sell_orders:
            return None, None
        return max(depth.buy_orders), min(depth.sell_orders)

    def _mid(self, depth: OrderDepth) -> float:
        bid, ask = self._best_bid_ask(depth)
        if bid is None:
            return None
        if bid < ask:
            return 0.5 * (bid + ask)
        return float(bid)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _smile_iv(self, S: float, K: float, T: float) -> float:
        if S <= 0 or T <= 0:
            return max(0.05, self.SMILE_C)
        m = math.log(K / S) / math.sqrt(T)
        iv = self.SMILE_A * m * m + self.SMILE_B * m + self.SMILE_C
        return min(1.8, max(0.05, iv))

    def _bs_call_delta_vega(self, S: float, K: float, T: float, sigma: float) -> Tuple[float, float, float]:
        if T <= 1e-6:
            intrinsic = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
            return intrinsic, delta, 0.0

        sigma = max(1e-6, sigma)
        root_t = math.sqrt(T)
        d1 = (math.log(max(S, 1e-9) / K) + 0.5 * sigma * sigma * T) / (sigma * root_t)
        d2 = d1 - sigma * root_t
        call = S * self._norm_cdf(d1) - K * self._norm_cdf(d2)
        delta = self._norm_cdf(d1)
        vega = S * self._norm_pdf(d1) * root_t
        return call, delta, vega

    def _remaining_buy(self, product: str, position: int) -> int:
        return max(0, self.LIMITS[product] - position)

    def _remaining_sell(self, product: str, position: int) -> int:
        return max(0, self.LIMITS[product] + position)

    def _take_underlying_mr(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        deviation: float,
    ) -> List[Order]:
        orders: List[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        
        if position >= self.UNDERLYING_SOFT_LIMIT and deviation <= 0:
            return orders
        if position <= -self.UNDERLYING_SOFT_LIMIT and deviation >= 0:
            return orders

        if deviation > self.UNDERLYING_THR:
            
            remain = min(self.UNDERLYING_MAX_TAKE, self._remaining_sell(product, position))
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if remain <= 0:
                    break
                avail = depth.buy_orders[bid]
                qty = min(remain, avail)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    remain -= qty

        elif deviation < -self.UNDERLYING_THR:
            
            remain = min(self.UNDERLYING_MAX_TAKE, self._remaining_buy(product, position))
            for ask in sorted(depth.sell_orders.keys()):
                if remain <= 0:
                    break
                avail = -depth.sell_orders[ask]
                qty = min(remain, avail)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    remain -= qty

        return orders

    def _quote_hydrogel(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None:
            return orders

        if ask - bid > 1:
            bid_px = bid + 1
            ask_px = ask - 1
        else:
            bid_px = bid
            ask_px = ask

        
        skew = int(round(position / 60.0))
        bid_px -= max(0, skew)
        ask_px -= min(0, skew)

        buy_qty = min(self.HYDRO_QUOTE_SIZE, self._remaining_buy("HYDROGEL_PACK", position))
        sell_qty = min(self.HYDRO_QUOTE_SIZE, self._remaining_sell("HYDROGEL_PACK", position))

        if buy_qty > 0:
            orders.append(Order("HYDROGEL_PACK", bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order("HYDROGEL_PACK", ask_px, -sell_qty))
        return orders

    def _option_scalping_orders(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        underlying_mid: float,
        tte_years: float,
        data: dict,
    ) -> List[Order]:
        orders: List[Order] = []
        bid, ask = self._best_bid_ask(depth)
        if bid is None or underlying_mid is None:
            return orders

        wall_mid = 0.5 * (bid + ask)
        strike = self.STRIKES[product]

        iv = self._smile_iv(underlying_mid, strike, tte_years)
        theo, _delta, vega = self._bs_call_delta_vega(underlying_mid, strike, tte_years, iv)
        theo_diff = wall_mid - theo

        mean_diff = self._ema(data, f"{product}:theo_diff_ema", self.THEO_NORM_WINDOW, theo_diff)
        dev = theo_diff - mean_diff
        avg_abs_dev = self._ema(data, f"{product}:dev_abs_ema", self.IV_SCALPING_WINDOW, abs(dev))

        
        if avg_abs_dev < self.IV_SCALPING_ON_THRESHOLD:
            
            if position > 0:
                qty = min(position, min(self.MAX_OPTION_TAKE, depth.buy_orders.get(bid, 0)))
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
            elif position < 0:
                qty = min(-position, min(self.MAX_OPTION_TAKE, -depth.sell_orders.get(ask, 0)))
                if qty > 0:
                    orders.append(Order(product, ask, qty))
            return orders

        low_vega_adj = self.LOW_VEGA_ADJ if vega <= 1.0 else 0.0
        open_thr = self.OPEN_Z * max(0.5, avg_abs_dev) + low_vega_adj
        close_thr = self.CLOSE_Z * max(0.5, avg_abs_dev)

        
        edge_sell = bid - theo - mean_diff
        edge_buy = theo - ask + mean_diff

        
        if position > 0 and edge_sell >= close_thr:
            qty = min(position, min(self.MAX_OPTION_TAKE, depth.buy_orders.get(bid, 0)))
            if qty > 0:
                orders.append(Order(product, bid, -qty))

        elif position < 0 and edge_buy >= close_thr:
            qty = min(-position, min(self.MAX_OPTION_TAKE, -depth.sell_orders.get(ask, 0)))
            if qty > 0:
                orders.append(Order(product, ask, qty))

        
        if edge_sell >= open_thr:
            
            qty = min(self.MAX_OPTION_TAKE, self._remaining_sell(product, position))
            qty = min(qty, depth.buy_orders.get(bid, 0))
            if qty > 0:
                orders.append(Order(product, bid, -qty))

        elif edge_buy >= open_thr:
            
            qty = min(self.MAX_OPTION_TAKE, self._remaining_buy(product, position))
            qty = min(qty, -depth.sell_orders.get(ask, 0))
            if qty > 0:
                orders.append(Order(product, ask, qty))

        return orders

    def run(self, state: TradingState):
        data = self._load_data(state)
        result: Dict[str, List[Order]] = {}

        vel_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vel_mid = self._mid(vel_depth) if vel_depth else None

        
        frac_day = max(0.0, min(1.0, state.timestamp / 1_000_000.0))
        tte_days = max(0.2, self.ROUND3_START_TTE_DAYS - frac_day)
        tte_years = tte_days / self.DAYS_PER_YEAR

        
        if vel_depth and vel_mid is not None:
            ema_u = self._ema(data, "VELVET:ema", self.UNDERLYING_EMA_WINDOW, vel_mid)
            u_dev = vel_mid - ema_u
            pos_u = int(state.position.get("VELVETFRUIT_EXTRACT", 0))
            result["VELVETFRUIT_EXTRACT"] = self._take_underlying_mr(
                "VELVETFRUIT_EXTRACT",
                vel_depth,
                pos_u,
                u_dev,
            )

        
        for sym in self.OPTION_SYMBOLS:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue

            if sym not in self.ACTIVE_SCALPING_SYMBOLS:
                
                continue

            pos = int(state.position.get(sym, 0))

            
            result[sym] = self._option_scalping_orders(
                sym,
                depth,
                pos,
                vel_mid,
                tte_years,
                data,
            )

        
        hydro_depth = state.order_depths.get("HYDROGEL_PACK")
        if hydro_depth:
            hydro_pos = int(state.position.get("HYDROGEL_PACK", 0))
            result["HYDROGEL_PACK"] = self._quote_hydrogel(hydro_depth, hydro_pos)

        trader_data = self._dump_data(data)
        conversions = 0
        return result, conversions, trader_data
