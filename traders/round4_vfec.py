from __future__ import annotations
from collections import deque
import math
from datamodel import Order, TradingState





































VFE_PRODUCT    = "VELVETFRUIT_EXTRACT"
ALL_STRIKES    = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_LIMIT   = 300          
ORDER_SIZE     = 10           



TTE_DAYS_START = 4
TICKS_PER_DAY  = 10_000
TRADING_DAYS   = 252          


VOL_WINDOW     = 200          
VOL_FALLBACK   = 1.0          


MIN_EDGE       = 2.0          
EDGE_SPREAD_MULT = 0.6        
EXIT_ZSCORE    = 1.5          


DELTA_LIMIT    = 50.0         


TV_SEED = {
    4000: 0.0, 4500: 0.0, 5000: 6.8,  5100: 21.6, 5200: 51.0,
    5300: 46.1, 5400: 18.5, 5500: 8.1, 6000: 0.5,  6500: 0.5,
}





def _norm_cdf(x: float) -> float:
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    return 0.5 + sign * (0.5 - math.exp(-0.5*x*x) / math.sqrt(2*math.pi) * poly)

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    
    if T <= 0:
        return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    return _norm_cdf(d1)

def implied_vol(S: float, K: float, T: float, market_price: float,
                lo: float = 0.01, hi: float = 20.0, iters: int = 40) -> float:
    
    if T <= 0 or market_price <= max(0.0, S - K):
        return VOL_FALLBACK
    for _ in range(iters):
        mid = (lo + hi) / 2
        if bs_call_price(S, K, T, mid) > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return (lo + hi) / 2


class VoucherStrategy:

    def __init__(self, spot_ref: "SpotReference"):
        self._spot = spot_ref        
        self._tick = 0

        
        self._iv_ema:  dict[int, float]  = {K: VOL_FALLBACK for K in ALL_STRIKES}
        self._tv_hist: dict[int, deque]  = {K: deque(maxlen=200) for K in ALL_STRIKES}
        self._spread_ema: dict[int, float | None] = {K: None for K in ALL_STRIKES}

    def _tte(self) -> float:
        
        days_elapsed = self._tick / TICKS_PER_DAY
        days_left = max(0.0, TTE_DAYS_START - days_elapsed)
        return days_left / TRADING_DAYS

    def _net_delta(self, positions: dict) -> float:
        S   = self._spot.mid
        T   = self._tte()
        net = 0.0
        for K in ALL_STRIKES:
            pos = positions.get(f"VEV_{K}", 0)
            if pos == 0: continue
            sig = self._iv_ema[K]
            net += pos * bs_call_delta(S, K, T, sig)
        return net

    def trade(self, order_depths: dict, positions: dict) -> dict:
        self._tick += 1
        result: dict = {}

        S = self._spot.mid
        if S <= 0: return result
        T = self._tte()

        net_delta = self._net_delta(positions)

        for K in ALL_STRIKES:
            product = f"VEV_{K}"
            od = order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid_opt  = (best_bid + best_ask) / 2.0
            spread   = best_ask - best_bid

            
            se = self._spread_ema[K]
            self._spread_ema[K] = float(spread) if se is None else 0.85*se + 0.15*spread

            
            iv = implied_vol(S, K, T, mid_opt)
            self._iv_ema[K] = 0.90*self._iv_ema[K] + 0.10*iv

            
            fair = bs_call_price(S, K, T, self._iv_ema[K])

            
            tv_obs = mid_opt - max(0.0, S - K)
            if tv_obs >= 0:
                self._tv_hist[K].append(tv_obs)

            
            edge = max(MIN_EDGE, (self._spread_ema[K] or 2.0) * EDGE_SPREAD_MULT)

            cur_pos = positions.get(product, 0)
            orders  = []
            delta_K = bs_call_delta(S, K, T, self._iv_ema[K])

            
            if (best_ask <= fair - edge
                    and cur_pos < OPTION_LIMIT
                    and net_delta + delta_K <= DELTA_LIMIT):
                qty = min(-od.sell_orders[best_ask], OPTION_LIMIT - cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    net_delta += qty * delta_K

            
            elif (best_bid >= fair + edge
                    and cur_pos > -OPTION_LIMIT
                    and net_delta - delta_K >= -DELTA_LIMIT):
                qty = min(od.buy_orders[best_bid], OPTION_LIMIT + cur_pos, ORDER_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    net_delta -= qty * delta_K

            
            elif cur_pos != 0 and len(self._tv_hist[K]) >= 20:
                tv_arr = list(self._tv_hist[K])
                n   = len(tv_arr)
                mu  = sum(tv_arr) / n
                std = math.sqrt(sum((x-mu)**2 for x in tv_arr) / n)
                tv_now = mid_opt - max(0.0, S - K)
                z = (tv_now - mu) / std if std > 1e-6 else 0.0

                if cur_pos > 0 and z > EXIT_ZSCORE:
                    
                    qty = min(cur_pos, ORDER_SIZE, od.buy_orders.get(best_bid, 0))
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                elif cur_pos < 0 and z < -EXIT_ZSCORE:
                    qty = min(-cur_pos, ORDER_SIZE, -od.sell_orders.get(best_ask, 0))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))

            if orders:
                result[product] = orders

        return result





class SpotReference:
    def __init__(self):
        self.mid = 5250.0   
        self._ema = None

    def update(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:   m = (max(b) + min(a)) / 2.0
        elif b:       m = float(max(b))
        elif a:       m = float(min(a))
        else:         return self.mid
        self._ema = m if self._ema is None else 0.92*self._ema + 0.08*m
        self.mid  = self._ema
        return self.mid





class Trader:
    def __init__(self):
        self._spot    = SpotReference()
        self._voucher = VoucherStrategy(self._spot)

    def run(self, state: TradingState):
        result = {}
        od  = state.order_depths
        pos = state.position

        
        if VFE_PRODUCT in od:
            self._spot.update(od[VFE_PRODUCT])

        result.update(self._voucher.trade(od, pos))

        return result, 0, ""
