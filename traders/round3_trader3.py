from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from datamodel import Order, TradingState






def _ncdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = t * (0.319381530 + t * (-0.356563782
          + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    v = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * p
    return v if x >= 0 else 1.0 - v

def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9: return max(0.0, S - K)
    if S <= 0 or sigma <= 0: return max(0.0, S - K)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9: return 1.0 if S > K else 0.0
    if S <= 0 or sigma <= 0: return 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return _ncdf(d1)

def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or S <= 0 or sigma <= 0: return 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return _npdf(d1) / (S * sq)

def bs_theta(S: float, K: float, T: float, sigma: float) -> float:
    """Theta per day."""
    if T <= 1e-9 or S <= 0 or sigma <= 0: return 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return -(S * _npdf(d1) * sigma / (2.0 * math.sqrt(T)))






SIGMA_IMPLIED    = 0.0130    
SIGMA_REALIZED   = 0.0216    
TICKS_PER_DAY    = 10_000

ALL_STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
BELLY_STRIKES    = [5000, 5100, 5200, 5300, 5400, 5500]


ACTIVATION_BAND      = 0.030   
DEACTIVATION_BAND    = 0.055   
DAILY_EDGE_MIN       = 250.0   
EDGE_BUFFER          = 1.10    
EXPIRY_BAND_TIGHTEN  = 0.020   


HEDGE_FREQ_NORMAL    = 50      
HEDGE_FREQ_EXPIRY    = 20      
DELTA_REHEDGE_MIN    = 0.03    
VFE_LIMIT            = 200     
HYDROGEL_LIMIT       = 200
OPTION_LIMIT         = 300


VOL_WINDOW           = 200     
VOL_MIN_OBS          = 50      


VFE_MM_CLIP          = 15
VFE_MM_SKEW          = 0.06
VFE_TAKE_EDGE        = 1


HYDRO_CLIP           = 30
HYDRO_SKEW           = 0.10
HYDRO_TAKE_EDGE      = 2






class RollingVolEstimator:
    """
    Estimates per-day realized volatility from the last `window` tick log-returns.
    Annualises to per-day units (×√TICKS_PER_DAY) so it is comparable to sigma_implied.
    """

    def __init__(self, window: int = VOL_WINDOW):
        self._log_returns: deque = deque(maxlen=window)
        self._last_price: Optional[float] = None

    def update(self, price: float) -> None:
        if self._last_price is not None and self._last_price > 0 and price > 0:
            lr = math.log(price / self._last_price)
            self._log_returns.append(lr)
        self._last_price = price

    @property
    def sigma(self) -> float:
        """Per-day realized vol. Returns SIGMA_REALIZED if insufficient data."""
        if len(self._log_returns) < VOL_MIN_OBS:
            return SIGMA_REALIZED
        arr = list(self._log_returns)
        mean = sum(arr) / len(arr)
        var  = sum((r - mean) ** 2 for r in arr) / (len(arr) - 1)
        return math.sqrt(var * TICKS_PER_DAY)   

    @property
    def ready(self) -> bool:
        return len(self._log_returns) >= VOL_MIN_OBS






def projected_daily_edge(S: float, K: int, T: float,
                         sigma_i: float, sigma_r: float,
                         position: int) -> float:
    """
    Expected net daily PnL for a delta-hedged long option position.
    = 0.5 × Γ × S² × (σ_r² − σ_i²) × TICKS_PER_DAY  +  θ × position
    """
    g = bs_gamma(S, K, T, sigma_i)
    t = bs_theta(S, K, T, sigma_i)
    gamma_edge = 0.5 * g * S ** 2 * (sigma_r ** 2 - sigma_i ** 2) * TICKS_PER_DAY
    theta_cost = t  
    return (gamma_edge + theta_cost) * position


def hedge_delta_required(S: float, K: int, T: float,
                         sigma_i: float, position: int) -> float:
    """Net VFE units needed to delta-hedge `position` calls at strike K."""
    return -position * bs_delta(S, K, T, sigma_i)






@dataclass
class StrikeBook:
    """Tracks state for a single active option strike."""
    K: int
    position: int = 0           
    last_delta: float = 0.0     
    is_active: bool = False
    activation_tick: int = 0

    def moneyness(self, S: float) -> float:
        return math.log(S / self.K) if S > 0 else 0.0

    def should_deactivate(self, S: float, T: float,
                          sigma_r: float, band: float) -> bool:
        """True if this strike has drifted too far from ATM or lost its edge."""
        mono = abs(self.moneyness(S))
        edge = projected_daily_edge(S, self.K, T, SIGMA_IMPLIED, sigma_r, self.position)
        return mono > band or edge < DAILY_EDGE_MIN * 0.5

    def delta_gap(self, S: float, T: float) -> float:
        """Current delta minus last hedged delta (triggers rehedge if large)."""
        current_delta = bs_delta(S, self.K, T, SIGMA_IMPLIED)
        return abs(current_delta - self.last_delta)






class HydrogelStrategy:
    PRODUCT = "HYDROGEL_PACK"
    DEFAULT_FAIR = 9990.0

    def __init__(self):
        self._fair: float = self.DEFAULT_FAIR

    def _update_fair(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:
            mid = (max(b) + min(a)) / 2.0
        elif b:
            mid = float(max(b))
        elif a:
            mid = float(min(a))
        else:
            return self._fair
        self._fair = 0.85 * self._fair + 0.15 * mid
        return self._fair

    def trade(self, od, pos: int) -> List[Order]:
        fair = self._update_fair(od)
        orders: List[Order] = []
        lim = HYDROGEL_LIMIT

        
        for ask in sorted(od.sell_orders):
            v = -od.sell_orders[ask]
            if ask <= fair - HYDRO_TAKE_EDGE and pos < lim:
                q = min(v, lim - pos)
                orders.append(Order(self.PRODUCT, ask, q))
                pos += q
        for bid in sorted(od.buy_orders, reverse=True):
            v = od.buy_orders[bid]
            if bid >= fair + HYDRO_TAKE_EDGE and pos > -lim:
                q = min(v, lim + pos)
                orders.append(Order(self.PRODUCT, bid, -q))
                pos -= q

        
        skew    = int(round(HYDRO_SKEW * pos))
        bb      = max(od.buy_orders) if od.buy_orders  else int(fair) - 8
        ba      = min(od.sell_orders) if od.sell_orders else int(fair) + 8
        bid_px  = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px  = max(ba - 1 - skew, int(math.ceil(fair))  + 1)
        if bid_px >= ask_px:
            bid_px, ask_px = int(math.floor(fair)) - 1, int(math.ceil(fair)) + 1
        bq = min(HYDRO_CLIP, max(0, lim - pos))
        aq = min(HYDRO_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders






class VFEStrategy:
    PRODUCT    = "VELVETFRUIT_EXTRACT"
    DEFAULT_MID = 5250.0

    def __init__(self):
        self._ema: float = self.DEFAULT_MID
        self.hedge_target: Optional[int] = None   
        self.vol_estimator = RollingVolEstimator()

    def mid(self) -> float:
        return self._ema

    def _update_ema(self, od) -> float:
        b, a = od.buy_orders, od.sell_orders
        if b and a:   m = (max(b) + min(a)) / 2.0
        elif b:       m = float(max(b))
        elif a:       m = float(min(a))
        else:         return self._ema
        self.vol_estimator.update(m)
        self._ema = 0.90 * self._ema + 0.10 * m
        return self._ema

    def realized_vol(self) -> float:
        return self.vol_estimator.sigma

    def trade(self, od, pos: int) -> List[Order]:
        fair = self._update_ema(od)
        orders: List[Order] = []
        lim = VFE_LIMIT

        
        if self.hedge_target is not None and abs(self.hedge_target - pos) > 3:
            gap = self.hedge_target - pos
            if gap > 0 and od.sell_orders:
                for ask in sorted(od.sell_orders):
                    if gap <= 0: break
                    q = min(-od.sell_orders[ask], gap, lim - pos)
                    if q > 0:
                        orders.append(Order(self.PRODUCT, ask, q))
                        pos += q; gap -= q
            elif gap < 0 and od.buy_orders:
                for bid in sorted(od.buy_orders, reverse=True):
                    if gap >= 0: break
                    q = min(od.buy_orders[bid], -gap, lim + pos)
                    if q > 0:
                        orders.append(Order(self.PRODUCT, bid, -q))
                        pos -= q; gap += q

        
        skew   = int(round(VFE_MM_SKEW * pos))
        bb     = max(od.buy_orders)  if od.buy_orders  else int(fair) - 2
        ba     = min(od.sell_orders) if od.sell_orders else int(fair) + 2
        bid_px = min(bb + 1 - skew, int(math.floor(fair)) - 1)
        ask_px = max(ba - 1 - skew, int(math.ceil(fair))  + 1)
        if bid_px >= ask_px:
            bid_px, ask_px = int(math.floor(fair)) - 1, int(math.ceil(fair)) + 1
        bq = min(VFE_MM_CLIP, max(0, lim - pos))
        aq = min(VFE_MM_CLIP, max(0, lim + pos))
        if bq: orders.append(Order(self.PRODUCT, bid_px,  bq))
        if aq: orders.append(Order(self.PRODUCT, ask_px, -aq))
        return orders






class OptionsDeskStrategy:
    """
    Core algorithm.  Manages a book of active option strikes.

    PHASE LOGIC (run every tick):
    ─────────────────────────────
    1. Estimate current TTE and spot.
    2. For every belly strike, compute moneyness and projected daily edge.
    3. Phase 1 bootstrap: if no strikes active yet, activate the single
       nearest-to-ATM strike immediately (no vol-edge gate — we're in early).
    4. Phase 2 expansion: for inactive strikes that enter ACTIVATION_BAND
       and pass the edge/vol gate, activate them if VFE budget allows.
    5. Phase 2 contraction: for active strikes outside DEACTIVATION_BAND
       or with insufficient edge, close them (sell back options, cancel hedge).
    6. Phase 3 (TTE < 2): tighten the band, focus on single ATM strike,
       increase hedge frequency.
    7. Compute target delta for each active strike → sum → write hedge_target
       to VFE strategy.
    8. Generate option orders: passive bid at best_bid+1 for target entries;
       unwind via passive ask at best_ask-1 for exits.
    """

    ROUND3_START_TTE = 5.0

    def __init__(self, vfe: VFEStrategy):
        self.vfe = vfe
        self.books: Dict[int, StrikeBook] = {K: StrikeBook(K=K) for K in ALL_STRIKES}
        self._bootstrapped = False
        self._tick_count = 0

    def _tte(self, timestamp: int) -> float:
        return max(0.001, self.ROUND3_START_TTE - timestamp / TICKS_PER_DAY)

    def _hedge_freq(self, T: float) -> int:
        return HEDGE_FREQ_EXPIRY if T < 2.0 else HEDGE_FREQ_NORMAL

    def _activation_band(self, T: float) -> float:
        return EXPIRY_BAND_TIGHTEN if T < 2.0 else ACTIVATION_BAND

    def _deactivation_band(self, T: float) -> float:
        """Slightly wider than activation to create hysteresis."""
        return self._activation_band(T) * 1.8

    def _active_delta_budget_used(self, S: float, T: float,
                                  exclude_K: Optional[int] = None) -> float:
        """Sum of |hedge_units| for all active strikes (except exclude_K)."""
        total = 0.0
        for K, bk in self.books.items():
            if not bk.is_active or K == exclude_K: continue
            total += abs(hedge_delta_required(S, K, T, SIGMA_IMPLIED, bk.position))
        return total

    def _vfe_budget_remaining(self, S: float, T: float,
                               exclude_K: Optional[int] = None) -> float:
        used = self._active_delta_budget_used(S, T, exclude_K)
        return VFE_LIMIT - used

    def _activate_strike(self, K: int, S: float, T: float,
                          timestamp: int) -> None:
        bk = self.books[K]
        bk.is_active = True
        bk.position  = OPTION_LIMIT   
        bk.last_delta = bs_delta(S, K, T, SIGMA_IMPLIED)
        bk.activation_tick = timestamp

    def _deactivate_strike(self, K: int) -> None:
        bk = self.books[K]
        bk.is_active = False
        bk.position  = 0
        bk.last_delta = 0.0

    def trade(self, order_depths: Dict, positions: Dict[str, int],
              timestamp: int) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        self._tick_count += 1

        T = self._tte(timestamp)
        S = self.vfe.mid()
        if S <= 0:
            return result

        sigma_r      = self.vfe.realized_vol()
        act_band     = self._activation_band(T)
        deact_band   = self._deactivation_band(T)
        hedge_freq   = self._hedge_freq(T)
        vol_edge_ok  = sigma_r >= SIGMA_IMPLIED * EDGE_BUFFER

        
        active_count = sum(1 for bk in self.books.values() if bk.is_active)
        if active_count == 0:
            
            best_K = min(BELLY_STRIKES, key=lambda k: abs(math.log(S / k)))
            self._activate_strike(best_K, S, T, timestamp)

        
        
        for K in BELLY_STRIKES:
            bk = self.books[K]
            if not bk.is_active: continue
            mono = abs(bk.moneyness(S))
            edge = projected_daily_edge(S, K, T, SIGMA_IMPLIED, sigma_r, bk.position)
            if mono > deact_band and edge < DAILY_EDGE_MIN * 0.3:
                self._deactivate_strike(K)

        
        for K in BELLY_STRIKES:
            bk = self.books[K]
            if bk.is_active: continue
            mono = abs(bk.moneyness(S))
            if mono > act_band: continue
            
            edge = projected_daily_edge(S, K, T, SIGMA_IMPLIED, sigma_r, OPTION_LIMIT)
            if edge < DAILY_EDGE_MIN: continue
            
            if self._tick_count > 1 and not vol_edge_ok: continue
            
            new_hedge = abs(hedge_delta_required(S, K, T, SIGMA_IMPLIED, OPTION_LIMIT))
            if self._vfe_budget_remaining(S, T) >= new_hedge * 0.8:
                self._activate_strike(K, S, T, timestamp)

        
        for K in BELLY_STRIKES:
            bk = self.books[K]
            product = f"VEV_{K}"
            od = order_depths.get(product)
            if od is None: continue
            current_pos = positions.get(product, 0)
            orders: List[Order] = []

            if bk.is_active:
                
                if current_pos < bk.position and od.buy_orders:
                    target_buy = bk.position - current_pos
                    
                    fair_v = bs_price(S, K, T, SIGMA_IMPLIED)
                    for ask in sorted(od.sell_orders):
                        if current_pos >= bk.position: break
                        if ask < fair_v - 0.5:
                            q = min(-od.sell_orders[ask], target_buy,
                                    OPTION_LIMIT - current_pos)
                            if q > 0:
                                orders.append(Order(product, ask, q))
                                current_pos += q; target_buy -= q
                    if target_buy > 0 and od.buy_orders:
                        passive_bid = max(od.buy_orders) + 1
                        if passive_bid < fair_v - 0.3:
                            orders.append(Order(product, passive_bid,
                                                min(target_buy, 25)))
            else:
                
                if current_pos > 0 and od.buy_orders:
                    fair_v = bs_price(S, K, T, SIGMA_IMPLIED)
                    for bid in sorted(od.buy_orders, reverse=True):
                        if current_pos <= 0: break
                        q = min(od.buy_orders[bid], current_pos)
                        if q > 0:
                            orders.append(Order(product, bid, -q))
                            current_pos -= q
                    if current_pos > 0 and od.sell_orders:
                        passive_ask = min(od.sell_orders) - 1
                        if passive_ask > fair_v + 0.3:
                            orders.append(Order(product, passive_ask,
                                                -min(current_pos, 25)))

            if orders:
                result[product] = orders

        
        
        total_delta = 0.0
        needs_hedge = False
        for K in BELLY_STRIKES:
            bk = self.books[K]
            if not bk.is_active: continue
            actual_pos = positions.get(f"VEV_{K}", 0)
            if actual_pos == 0: continue
            delta_now = bs_delta(S, K, T, SIGMA_IMPLIED)
            total_delta += actual_pos * delta_now
            
            if bk.delta_gap(S, T) >= DELTA_REHEDGE_MIN:
                needs_hedge = True
                bk.last_delta = delta_now

        
        if total_delta != 0.0 and (timestamp % hedge_freq == 0 or needs_hedge):
            raw_target = -int(round(total_delta))
            self.vfe.hedge_target = max(-VFE_LIMIT, min(VFE_LIMIT, raw_target))
        elif total_delta == 0.0:
            self.vfe.hedge_target = None

        return result






class Trader:
    """
    Round 3 "Gloves Off" — Dynamic Gamma Scalping  (v4)

    Products:
      HYDROGEL_PACK                  → market-making
      VELVETFRUIT_EXTRACT            → market-making + delta hedge conduit
      VEV_4000 … VEV_6500            → dynamic gamma scalping

    Strategy phases:
      Phase 1: Single ATM strike activated immediately (K nearest to spot)
      Phase 2: New strikes activated as spot drifts, gated on vol edge
               and VFE budget; old strikes deactivated as they drift OTM
      Phase 3: TTE < 2 days — concentrate on tightest ATM, hedge every 20 ticks
    """

    def __init__(self):
        self.hydrogel = HydrogelStrategy()
        self.vfe      = VFEStrategy()
        self.options  = OptionsDeskStrategy(vfe=self.vfe)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        od  = state.order_depths
        pos = state.position
        ts  = state.timestamp

        
        result.update(self.options.trade(od, pos, ts))

        
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

        return result, 0, ""
