from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from collections import defaultdict
import math


class VoucherFairValueCalculator:    
    def __init__(self, sigma: float = 0.24):
        self.sigma = sigma
    
    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def bs_call(self, S: float, K: float, T: float) -> float:
        """
        Black–Scholes call price.
        T in years (e.g., 5/365 for 5 days).
        """
        if T <= 0.001:
            return max(S - K, 0.0)
        
        sigma = max(self.sigma, 1e-6)
        S = max(S, 1e-6)
        rt = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * rt)
        d2 = d1 - sigma * rt
        
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)


class Round3Trader:
    """
      - Market-making on delta-1 products (HYDROGEL_PACK, VELVETFRUIT_EXTRACT)
      - Mean-reversion short on ATM options (VEV_5200, VEV_5300)
      - Opportunistic long on near-OTM options (VEV_4500, VEV_5500)
    """
    
    POSITION_LIMITS = {
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
    
    VOUCHER_SYMBOLS = {
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
    
    MM_QUOTE_SIZE = 5
    MM_TICK_OFFSET = 1
    
    AGGRESSIVE_SHORT_SYMBOLS = {"VEV_5200", "VEV_5300"}
    CONSERVATIVE_LONG_SYMBOLS = {"VEV_4500", "VEV_5500"}
    SKIP_SYMBOLS = {"VEV_4000", "VEV_6000", "VEV_6500"}
    ATM_SHORT_SIZE = 20  
    NEARMOTM_LONG_SIZE = 8  
    
    def __init__(self):
        self.fair_calc = VoucherFairValueCalculator(sigma=0.24)
        self.position_trail = {}  
    
    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        
        orders_by_product: Dict[str, List[Order]] = {}
        
        for product, order_depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue
            
            position = int(state.position.get(product, 0))
            
            if product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
                orders_by_product[product] = self._trade_delta_one(
                    product, order_depth, position
                )
            else:
                
                orders_by_product[product] = self._trade_voucher(
                    product, order_depth, position, state
                )
        
        return orders_by_product, 0, ""
    
    def _micro_price(self, order_depth: OrderDepth) -> float:
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = max(1, order_depth.buy_orders[best_bid])
        ask_vol = max(1, -order_depth.sell_orders[best_ask])
        
        return (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)
    
    def _get_mid(self, order_depth: OrderDepth) -> float:
        
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid = (best_bid + best_ask) / 2.0
            if best_bid < best_ask:
                return mid
        
        micro = self._micro_price(order_depth)
        if micro is not None:
            return micro
        
        if order_depth.buy_orders:
            return float(max(order_depth.buy_orders))
        if order_depth.sell_orders:
            return float(min(order_depth.sell_orders))
        
        return None
    
    def _trade_delta_one(
        self, product: str, order_depth: OrderDepth, position: int
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        
        mid = self._get_mid(order_depth)
        if mid is None:
            return []
        
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []
        
        
        bid_price = int(round(mid - self.MM_TICK_OFFSET))
        ask_price = int(round(mid + self.MM_TICK_OFFSET))
        
        
        buy_size = min(self.MM_QUOTE_SIZE, max(1, limit - position))
        sell_size = min(self.MM_QUOTE_SIZE, max(1, limit + position))
        
        
        if position > 80:
            ask_price = int(round(mid))
            sell_size = min(max(position - 40, self.MM_QUOTE_SIZE), limit + position)
        elif position < -80:
            bid_price = int(round(mid))
            buy_size = min(max(abs(position) - 40, self.MM_QUOTE_SIZE), limit - position)
        
        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
        
        return orders
    
    def _trade_voucher(
        self, product: str, order_depth: OrderDepth, position: int, state: TradingState
    ) -> List[Order]:
        """
        Trade vouchers (options) using mean reversion and premium collection.
        
        Strategy:
          - ATM (VEV_5200, VEV_5300): aggressive short (they're underpriced)
          - Near-OTM (VEV_4500, VEV_5500): conservative long (better value)
          - Deep OTM: skip or small passive
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        
        mid = self._get_mid(order_depth)
        if mid is None:
            return []
        
        
        underlying = state.order_depths.get("VELVETFRUIT_EXTRACT")
        underlying_mid = self._get_mid(underlying) if underlying else None
        K = self.VOUCHER_SYMBOLS.get(product)
        
        
        
        TTE_days = 5
        T = TTE_days / 365.0
        
        fair = None
        if underlying_mid is not None and K is not None:
            fair = self.fair_calc.bs_call(underlying_mid, K, T)
        
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        
        
        if product in self.AGGRESSIVE_SHORT_SYMBOLS:
            orders = self._aggressively_short_atm(product, order_depth, position, mid, limit)
        elif product in self.CONSERVATIVE_LONG_SYMBOLS:
            orders = self._conservatively_long_nearotm(product, order_depth, position, mid, limit)
        else:
            
            orders = self._passive_quote(product, order_depth, position, mid, limit)
        
        return orders
    
    def _aggressively_short_atm(
        self, product: str, order_depth: OrderDepth, position: int, mid: float, limit: int
    ) -> List[Order]:
        """
        Short ATM vouchers (VEV_5200, VEV_5300).
        
        These are consistently underpriced (mean mispricing ≈ -1.7).
        Sell at mid or mid+0.5, buy dips below mid-1, hold 1–2 ticks.
        """
        orders: List[Order] = []
        
        
        sell_price = int(round(mid + 0.5))
        max_short = limit + position
        sell_qty = min(self.ATM_SHORT_SIZE, max_short)
        
        if sell_qty > 0 and sell_price <= min(order_depth.sell_orders):
            
            orders.append(Order(product, sell_price, -sell_qty))
        
        
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        if best_bid is not None and best_bid <= mid - 1.5:
            buy_qty = min(self.ATM_SHORT_SIZE // 2, limit - position)
            if buy_qty > 0:
                orders.append(Order(product, int(round(best_bid)), buy_qty))
        
        return orders
    
    def _conservatively_long_nearotm(
        self, product: str, order_depth: OrderDepth, position: int, mid: float, limit: int
    ) -> List[Order]:
        orders: List[Order] = []
        
        
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        if best_bid is not None and mid - best_bid < 2:
            buy_qty = min(self.NEARMOTM_LONG_SIZE, limit - position)
            if buy_qty > 0:
                orders.append(Order(product, best_bid + 1, buy_qty))
        
        return orders
    
    def _passive_quote(
        self, product: str, order_depth: OrderDepth, position: int, mid: float, limit: int
    ) -> List[Order]:
        orders: List[Order] = []
        
        
        if abs(position) > 30:
            return []
        
        bid_price = int(round(mid - 2))
        ask_price = int(round(mid + 2))
        
        buy_qty = min(2, max(0, limit - position))
        sell_qty = min(2, max(0, limit + position))
        
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))
        
        return orders



Trader = Round3Trader



TraderClass = Round3Trader

def run(state: TradingState) -> (Dict[str, List[Order]], int, str):
    trader = Round3Trader()
    return trader.run(state)

