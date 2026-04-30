from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional
import math

from datamodel import Order, TradingState






def _norm_cdf(x: float) -> float:
    
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
              + t * (-0.356563782
              + t * (1.781477937
              + t * (-1.821255978
              + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return cdf if x >= 0.0 else 1.0 - cdf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    
    if T <= 1e-9:
        return max(0.0, S - K)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    d2 = d1 - sq
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return _norm_cdf(d1)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    
    if T <= 1e-9:
        return 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return _norm_pdf(d1) / (S * sq)






class HydrogelPackStrategy:
    """
    Pure market-making strategy for HYDROGEL_PACK.

    Observations:
      - Mean-reverts tightly around ~9 990 (σ ≈ 30 ticks)
      - Natural bid-ask spread ≈ 16 ticks (half-spread ≈ 8)
      - Position limit: 200

    We quote 1 tick inside the best bid/ask, skewed by inventory, and also
    take aggressively whenever price dislocates more than 1 tick from fair.
    """

    PRODUCT        = "HYDROGEL_PACK"
    POSITION_LIMIT = 200
    DEFAULT_FAIR   = 9990.0
    INVENTORY_SKEW = 0.08   
    HALF_SPREAD    = 7      

    def __init__(self):
        self.fair_anchor: Optional[float] = None

    
    def _observe_mid(self, od) -> Optional[float]:
        bids, asks = od.buy_orders, od.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        if bids:
            return float(max(bids))
        if asks:
            return float(min(asks))
        return None

    def _fair(self, od) -> float:
        mid = self._observe_mid(od)
        if self.fair_anchor is None:
            self.fair_anchor = mid if mid is not None else self.DEFAULT_FAIR
            return self.fair_anchor
        if mid is not None:
            self.fair_anchor = 0.85 * self.fair_anchor + 0.15 * mid
        return self.fair_anchor

    
    def _take(self, od, position: int, fair: float) -> List[Order]:
        orders = []
        limit = self.POSITION_LIMIT

        for ask in sorted(od.sell_orders):
            vol = -od.sell_orders[ask]
            qty = 0
            if ask <= fair - 1 and position < limit:
                qty = min(vol, limit - position)
            elif ask <= fair and position < 0:
                qty = min(vol, -position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, ask, qty))
                position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            vol = od.buy_orders[bid]
            qty = 0
            if bid >= fair + 1 and position > -limit:
                qty = min(vol, limit + position)
            elif bid >= fair and position > 0:
                qty = min(vol, position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bid, -qty))
                position -= qty

        return orders

    
    def _make(self, od, position: int, fair: float) -> List[Order]:
        orders = []
        limit = self.POSITION_LIMIT

        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half = max(1, (best_ask - best_bid) // 2)
        else:
            half = self.HALF_SPREAD

        skew      = int(round(self.INVENTORY_SKEW * position))
        bid_px    = int(round(fair - half - skew))
        ask_px    = int(round(fair + half - skew))

        if best_bid is not None:
            bid_px = max(bid_px, best_bid + 1 if best_bid < fair else best_bid)
        if best_ask is not None:
            ask_px = min(ask_px, best_ask - 1 if best_ask > fair else best_ask)

        if bid_px >= ask_px:
            bid_px, ask_px = int(round(fair)) - 1, int(round(fair)) + 1

        buy_qty  = max(0, limit - position)
        sell_qty = max(0, limit + position)

        if buy_qty  > 0: orders.append(Order(self.PRODUCT, bid_px,  buy_qty))
        if sell_qty > 0: orders.append(Order(self.PRODUCT, ask_px, -sell_qty))
        return orders

    
    def trade(self, od, position: int) -> List[Order]:
        fair = self._fair(od)
        take = self._take(od, position, fair)
        pos2 = position + sum(o.quantity for o in take)
        return take + self._make(od, pos2, fair)






class VelvetfruitExtractStrategy:
    """
    Market-making strategy for VELVETFRUIT_EXTRACT (VFE).

    Observations:
      - Mean-reverts around ~5 250, natural spread ≈ 5 ticks (half ≈ 2-3)
      - Realised daily vol ≈ 2.15 % – used to price options
      - Position limit: 200
      - This product ALSO acts as the delta hedge for the options portfolio.
        We accept residual delta from the options book and simply store the
        net VFE position target; the options strategy writes to self.hedge_qty
        each tick before this strategy runs.

    We quote 1 tick inside the spread with inventory skew.  When the options
    desk requests a hedge the target position overrides the soft limit.
    """

    PRODUCT        = "VELVETFRUIT_EXTRACT"
    POSITION_LIMIT = 200
    DEFAULT_FAIR   = 5250.0
    INVENTORY_SKEW = 0.06
    HALF_SPREAD    = 2

    def __init__(self):
        self.ema_mid: Optional[float] = None
        self.spread_ema: Optional[float] = None
        self.move_ema: float = 1.0
        self.last_mid: Optional[float] = None
        
        self.hedge_target: int = 0

    
    def _observe_mid(self, od) -> Optional[float]:
        bids, asks = od.buy_orders, od.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        if bids:  return float(max(bids))
        if asks:  return float(min(asks))
        return None

    def _update(self, od) -> Optional[float]:
        mid = self._observe_mid(od)
        if od.buy_orders and od.sell_orders:
            sp = min(od.sell_orders) - max(od.buy_orders)
            self.spread_ema = sp if self.spread_ema is None else 0.9 * self.spread_ema + 0.1 * sp
        if mid is not None:
            if self.last_mid is not None:
                self.move_ema = 0.9 * self.move_ema + 0.1 * abs(mid - self.last_mid)
            self.last_mid = mid
            self.ema_mid = mid if self.ema_mid is None else 0.9 * self.ema_mid + 0.1 * mid
        return mid

    def fair(self, od) -> float:
        self._update(od)
        return self.ema_mid if self.ema_mid is not None else self.DEFAULT_FAIR

    
    def _take(self, od, position: int, fair: float) -> List[Order]:
        orders = []
        limit = self.POSITION_LIMIT

        for ask in sorted(od.sell_orders):
            vol = -od.sell_orders[ask]
            qty = 0
            if ask <= fair - 1 and position < limit:
                qty = min(vol, limit - position)
            elif ask <= fair and position < 0:
                qty = min(vol, -position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, ask, qty))
                position += qty

        for bid in sorted(od.buy_orders, reverse=True):
            vol = od.buy_orders[bid]
            qty = 0
            if bid >= fair + 1 and position > -limit:
                qty = min(vol, limit + position)
            elif bid >= fair and position > 0:
                qty = min(vol, position)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bid, -qty))
                position -= qty

        return orders

    
    def _hedge_take(self, od, position: int, target: int) -> List[Order]:
        
        orders = []
        limit  = self.POSITION_LIMIT
        gap    = target - position           

        if gap > 0 and od.sell_orders:      
            for ask in sorted(od.sell_orders):
                if gap <= 0: break
                vol = -od.sell_orders[ask]
                qty = min(vol, gap, limit - position)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, ask, qty))
                    position += qty
                    gap      -= qty

        elif gap < 0 and od.buy_orders:     
            for bid in sorted(od.buy_orders, reverse=True):
                if gap >= 0: break
                vol = od.buy_orders[bid]
                qty = min(vol, -gap, limit + position)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, bid, -qty))
                    position -= qty
                    gap      += qty

        return orders

    
    def _make(self, od, position: int, fair: float) -> List[Order]:
        orders = []
        limit  = self.POSITION_LIMIT

        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            half = max(1, (best_ask - best_bid) // 2)
        else:
            half = self.HALF_SPREAD

        skew   = int(round(self.INVENTORY_SKEW * position))
        bid_px = int(round(fair - half - skew))
        ask_px = int(round(fair + half - skew))

        if best_bid is not None:
            bid_px = max(bid_px, best_bid + 1 if best_bid < fair else best_bid)
        if best_ask is not None:
            ask_px = min(ask_px, best_ask - 1 if best_ask > fair else best_ask)

        if bid_px >= ask_px:
            bid_px, ask_px = int(round(fair)) - 1, int(round(fair)) + 1

        buy_qty  = max(0, limit - position)
        sell_qty = max(0, limit + position)

        if buy_qty  > 0: orders.append(Order(self.PRODUCT, bid_px,  buy_qty))
        if sell_qty > 0: orders.append(Order(self.PRODUCT, ask_px, -sell_qty))
        return orders

    
    def trade(self, od, position: int) -> List[Order]:
        fair    = self.fair(od)
        target  = max(-self.POSITION_LIMIT,
                      min(self.POSITION_LIMIT, self.hedge_target))

        
        hedge_gap = abs(target - position)
        if hedge_gap > 5:
            hedge = self._hedge_take(od, position, target)
            pos2  = position + sum(o.quantity for o in hedge)
            return hedge + self._make(od, pos2, fair)
        else:
            take = self._take(od, position, fair)
            pos2 = position + sum(o.quantity for o in take)
            return take + self._make(od, pos2, fair)






class VelvetfruitVoucherStrategy:
    """
    Options vol-arb strategy for all 10 VEV vouchers.

    Core insight (from historical data analysis):
      - Realised daily vol of VFE ≈ 2.15 %
      - Market-implied vol ≈ 1.2 %  (computed via BS inversion on historical prices)
      - Options are systematically CHEAP  →  BUY options, earn realised-vol premium
      - Delta-hedge continuously via the VFE desk (writes to vfe_strategy.hedge_target)

    Round 3 TTE schedule:
      - Timestamp 0 of simulation  →  TTE = 5.0 days
      - Each tick is  1 / 10 000  of a day
      - TTE(t) = 5.0 - t / 10_000

    Option selection (delta-constrained to VFE position limit = 200):
      Max-edge LP solution → long VEV_5300 × 300, long VEV_5400 × 300
      Residual budget fills VEV_5200 or VEV_5500 depending on delta room.

    Near expiry (TTE < 1 day) we switch to intrinsic-value sniping only.
    """

    PRODUCT_PREFIX    = "VELVETFRUIT_EXTRACT_VOUCHER"
    POSITION_LIMIT    = 300
    SIGMA_REALIZED    = 0.0215   
    SIGMA_IMPLIED     = 0.0120   
    TICKS_PER_DAY     = 10_000
    ROUND3_START_TTE  = 5.0      
    PRICE_FLOOR       = 0.5      

    
    
    TARGET_LONGS = {
        5300: 300,
        5400: 300,
        5200: 115,   
    }

    def __init__(self, vfe_strategy: VelvetfruitExtractStrategy):
        self.vfe = vfe_strategy
        
        self.spot_ema: Optional[float] = None
        
        self.initial_buy_done: Dict[int, bool] = {k: False for k in self.TARGET_LONGS}

    
    def _tte(self, timestamp: int) -> float:
        return max(1e-6, self.ROUND3_START_TTE - timestamp / self.TICKS_PER_DAY)

    def _update_spot(self, vfe_od) -> float:
        bids, asks = vfe_od.buy_orders, vfe_od.sell_orders
        if bids and asks:
            mid = (max(bids) + min(asks)) / 2.0
        elif bids:
            mid = float(max(bids))
        elif asks:
            mid = float(min(asks))
        else:
            mid = self.spot_ema or 5250.0
        self.spot_ema = mid if self.spot_ema is None else 0.95 * self.spot_ema + 0.05 * mid
        return self.spot_ema

    def _fair_price(self, K: int, spot: float, tte: float) -> float:
        
        val = bs_call(spot, float(K), tte, self.SIGMA_REALIZED)
        return max(self.PRICE_FLOOR, val)

    
    def _trade_voucher(
        self,
        K: int,
        product: str,
        od,
        position: int,
        spot: float,
        tte: float,
    ) -> List[Order]:
        orders  = []
        limit   = self.POSITION_LIMIT
        fair    = self._fair_price(K, spot, tte)
        implied = bs_call(spot, float(K), tte, self.SIGMA_IMPLIED)
        mkt_implied = max(self.PRICE_FLOOR, implied)

        
        if od.sell_orders:
            for ask in sorted(od.sell_orders):
                vol = -od.sell_orders[ask]
                
                if ask < fair - 0.5 and position < limit:
                    qty = min(vol, limit - position)
                    if qty > 0:
                        orders.append(Order(product, ask, qty))
                        position += qty

        
        if od.buy_orders:
            for bid in sorted(od.buy_orders, reverse=True):
                vol = od.buy_orders[bid]
                
                if bid > fair + 0.5 and position > 0:
                    qty = min(vol, position)
                    if qty > 0:
                        orders.append(Order(product, bid, -qty))
                        position -= qty

        
        
        
        if position < limit:
            bid_px  = max(1, int(math.floor(mkt_implied)) - 1)
            buy_qty = min(10, limit - position)   
            if buy_qty > 0:
                orders.append(Order(product, bid_px, buy_qty))

        return orders

    
    def _compute_hedge_target(
        self,
        positions: Dict[str, int],
        spot: float,
        tte: float,
    ) -> int:
        
        total_delta = 0.0
        for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
            product = f"VELVETFRUIT_EXTRACT_VOUCHER_{K}"
            pos     = positions.get(product, 0)
            if pos != 0:
                d = bs_delta(spot, float(K), tte, self.SIGMA_REALIZED)
                total_delta += pos * d

        
        
        hedge = -int(round(total_delta))
        return max(-200, min(200, hedge))

    
    def trade(
        self,
        order_depths: Dict,
        positions: Dict[str, int],
        timestamp: int,
    ) -> Dict[str, List[Order]]:
        result  = {}
        tte     = self._tte(timestamp)
        vfe_od  = order_depths.get("VELVETFRUIT_EXTRACT")
        spot    = self._update_spot(vfe_od) if vfe_od else (self.spot_ema or 5250.0)

        for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
            product = f"VELVETFRUIT_EXTRACT_VOUCHER_{K}"
            od      = order_depths.get(product)
            if od is None:
                continue
            pos     = positions.get(product, 0)
            orders  = self._trade_voucher(K, product, od, pos, spot, tte)
            if orders:
                result[product] = orders

        
        hedge_target = self._compute_hedge_target(positions, spot, tte)
        self.vfe.hedge_target = hedge_target

        return result






class Trader:
    def __init__(self):
        self.hydrogel  = HydrogelPackStrategy()
        self.vfe       = VelvetfruitExtractStrategy()
        self.vouchers  = VelvetfruitVoucherStrategy(vfe_strategy=self.vfe)

    def run(self, state: TradingState):
        result  : Dict[str, List[Order]] = {}
        od      = state.order_depths
        pos     = state.position
        ts      = state.timestamp

        
        voucher_orders = self.vouchers.trade(od, pos, ts)
        result.update(voucher_orders)

        
        if "VELVETFRUIT_EXTRACT" in od:
            result["VELVETFRUIT_EXTRACT"] = self.vfe.trade(
                od["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0),
            )

        
        if "HYDROGEL_PACK" in od:
            result["HYDROGEL_PACK"] = self.hydrogel.trade(
                od["HYDROGEL_PACK"],
                pos.get("HYDROGEL_PACK", 0),
            )

        trader_data = ""
        conversions = 0
        return result, conversions, trader_data
