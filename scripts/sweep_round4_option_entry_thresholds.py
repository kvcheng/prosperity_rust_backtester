from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_TRADER = PROJECT_ROOT / "traders/round4_trader4.py"
TMP_TRADER = PROJECT_ROOT / "traders/sweep_tmp/_tmp_option_entry_sweep.py"


ADD_STRIKES: List[int] = [5100, 5200]


@dataclass(frozen=True)
class RunResult:
    total_pnl: float
    per_product: Dict[str, float]


def _patch_scalp_strikes(text: str, strikes: List[int]) -> str:
    return re.sub(
        r"^SCALP_STRIKES\s*=\s*\[[^\]]*\]\s*$",
        f"SCALP_STRIKES = {strikes}",
        text,
        flags=re.MULTILINE,
    )


def _parse_scalp_strikes(text: str) -> List[int]:
    m = re.search(r"^SCALP_STRIKES\s*=\s*\[([^\]]*)\]\s*$", text, flags=re.MULTILINE)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def _unique_keep_order(xs: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _patch_entry_thresh(text: str, entry_thresh: Dict[int, float]) -> str:
    # Replace the entire dict literal for ENTRY_THRESH.
    # Keep formatting simple/stable for regex safety.
    entries = "\n".join([f"    {k}: {entry_thresh[k]}," for k in sorted(entry_thresh)])
    replacement = "ENTRY_THRESH = {\n" + entries + "\n}"
    return re.sub(
        r"^ENTRY_THRESH\s*=\s*\{[\s\S]*?^\}\s*$",
        replacement,
        text,
        flags=re.MULTILINE,
    )


def _parse_summary(stdout: str) -> RunResult | None:
    per_product: Dict[str, float] = {}
    total_pnl: float | None = None

    # Parse top summary table: line that starts with TOTAL.
    for line in stdout.splitlines():
        s = line.strip()
        if not s.startswith("TOTAL"):
            continue
        parts = s.split()
        # Expected: TOTAL - 30000 <own_trades> <FINAL_PNL> -
        # We want parts[4] if it exists.
        if len(parts) >= 5:
            try:
                total_pnl = float(parts[4])
            except Exception:
                pass

    # Parse per-product table (PRODUCT ... TOTAL)
    # Example: VEV_5300 11049.00 7548.00 9000.00 27597.00
    prod_line = re.compile(r"^(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
    for line in stdout.splitlines():
        m = prod_line.match(line.strip())
        if not m:
            continue
        product = m.group(1)
        if product == "PRODUCT":
            continue
        try:
            per_product[product] = float(m.group(5))
        except Exception:
            continue

    if total_pnl is None:
        return None
    return RunResult(total_pnl=total_pnl, per_product=per_product)


def _run_backtest(entry_thresh: Dict[int, float], strikes: List[int]) -> RunResult | None:
    base_text = BASE_TRADER.read_text()
    text = _patch_scalp_strikes(base_text, strikes)
    text = _patch_entry_thresh(text, entry_thresh)
    TMP_TRADER.write_text(text)

    proc = subprocess.run(
        ["make", "-C", str(PROJECT_ROOT), "round4", f"TRADER=traders/sweep_tmp/{TMP_TRADER.name}"],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + "\n" + proc.stderr
    return _parse_summary(out)


def _coarse_values() -> List[float]:
    # Coarse grid, then refine around the winner.
    vals = [float(x) for x in range(0, 13)]  # 0..12
    vals += [15.0]
    return vals


def _refine_values(center: float) -> List[float]:
    # Fine grid around center, clipped to [0, 20]
    vals = [center + d for d in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)]
    clipped = sorted({max(0.0, min(20.0, round(v, 2))) for v in vals})
    return clipped


def _coordinate_optimize(
    traded_strikes: List[int],
    optimize_strikes: List[int],
    initial: Dict[int, float],
    passes: int = 2,
) -> Tuple[Dict[int, float], RunResult | None]:
    cur = dict(initial)
    for k in traded_strikes:
        cur.setdefault(k, 5.0)

    best_res = _run_backtest(cur, traded_strikes)
    if best_res is None:
        return cur, None

    for _ in range(passes):
        improved = False
        for k in optimize_strikes:
            base_val = float(cur[k])
            best_k_val = base_val
            best_k_res = best_res

            # Coarse sweep
            for v in _coarse_values():
                trial = dict(cur)
                trial[k] = v
                res = _run_backtest(trial, traded_strikes)
                if res is None:
                    continue
                if res.total_pnl > best_k_res.total_pnl:
                    best_k_res = res
                    best_k_val = v

            # Refine around winner
            for v in _refine_values(best_k_val):
                trial = dict(cur)
                trial[k] = v
                res = _run_backtest(trial, traded_strikes)
                if res is None:
                    continue
                if res.total_pnl > best_k_res.total_pnl:
                    best_k_res = res
                    best_k_val = v

            if best_k_val != base_val:
                cur[k] = best_k_val
                best_res = best_k_res
                improved = True
                print(f"strike {k}: {base_val} -> {best_k_val} | total={best_res.total_pnl:.2f}")

        if not improved:
            break

    return cur, best_res


def main() -> None:
    base_text = BASE_TRADER.read_text()
    base_traded = _parse_scalp_strikes(base_text)
    traded_strikes = _unique_keep_order(base_traded + ADD_STRIKES)
    optimize_strikes = list(ADD_STRIKES)

    # Default start point (uses existing if present in file; missing => 5.0)
    # Best-effort parse of current ENTRY_THRESH.
    initial: Dict[int, float] = {}
    m = re.search(r"^ENTRY_THRESH\s*=\s*\{([\s\S]*?)^\}\s*$", base_text, flags=re.MULTILINE)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line or line.startswith("#"):
                continue
            mm = re.match(r"^(\d+)\s*:\s*([0-9.]+)\s*$", line)
            if mm:
                initial[int(mm.group(1))] = float(mm.group(2))

    best_thresh, best_res = _coordinate_optimize(traded_strikes, optimize_strikes, initial, passes=2)
    if best_res is None:
        raise SystemExit("Failed to parse backtester output; no results.")

    print("\nBEST ENTRY_THRESH")
    for k in traded_strikes:
        print(f"  {k}: {best_thresh[k]}")
    print(f"\nTOTAL_PNL: {best_res.total_pnl:.2f}")
    print("\nPER_PRODUCT (TOTAL)")
    for p, v in sorted(best_res.per_product.items()):
        if p.startswith("VEV_") or p == "VELVETFRUIT_EXTRACT":
            print(f"  {p}: {v:.2f}")


if __name__ == "__main__":
    main()
