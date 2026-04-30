import itertools
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "traders" / "round2_trader.py"
TEMP_DIR = ROOT / "traders" / "tuned_tmp"
TEMP_DIR.mkdir(exist_ok=True)

BASE_TEXT = BASE_PATH.read_text()

GRID = {
    "Z_ENTRY": [1.10, 1.20, 1.30],
    "Z_EXIT": [0.25, 0.35, 0.45],
    "MAX_TURNOVER_PER_TICK": [24, 32, 40],
    "MIN_TARGET_SIZE": [10, 15, 22],
    "STRONG_BEAR_TARGET": [-30, -40, -55],
    "STRONG_BEAR_COOLDOWN": [4, 8, 12],
    "MAX_POST_SIZE": [12, 14, 18],
    "MAX_TAKE_PER_LEVEL": [14, 18, 22],
}

CANDIDATES = [
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.10, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.30, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.25, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.45, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 24, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 40, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 10, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 22, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -30, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -55, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 4, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 12, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 12, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 18, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 14},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.35, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 22},
    {"Z_ENTRY": 1.10, "Z_EXIT": 0.45, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.10, "Z_EXIT": 0.45, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 12, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.20, "Z_EXIT": 0.45, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 10, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
    {"Z_ENTRY": 1.00, "Z_EXIT": 0.45, "MAX_TURNOVER_PER_TICK": 32, "MIN_TARGET_SIZE": 15, "STRONG_BEAR_TARGET": -40, "STRONG_BEAR_COOLDOWN": 8, "MAX_POST_SIZE": 14, "MAX_TAKE_PER_LEVEL": 18},
]

PRODUCT_RE = re.compile(r"ASH_COATED_OSMIUM\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)")
TOTAL_RE = re.compile(r"TOTAL\s+-\s+\d+\s+\d+\s+([\d\.-]+)")


def patch_text(base_text: str, params: dict[str, float | int]) -> str:
    text = base_text
    for key, value in params.items():
        pattern = re.compile(rf"^(\s*){key}\s*=\s*.*$", re.MULTILINE)

        def repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{key} = {value}"

        text, n = pattern.subn(repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"Could not patch {key}")
    return text


def run_candidate(idx: int, params: dict[str, float | int]):
    trader_path = TEMP_DIR / f"round2_tuned_{idx}.py"
    trader_path.write_text(patch_text(BASE_TEXT, params))

    cmd = ["make", "round2", f"TRADER={trader_path}"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    output = proc.stdout + "\n" + proc.stderr

    product_match = PRODUCT_RE.search(output)
    total_match = TOTAL_RE.search(output)
    if proc.returncode != 0 or product_match is None or total_match is None:
        return {
            "idx": idx,
            "params": params,
            "ok": False,
            "rc": proc.returncode,
            "osmium": None,
            "total": None,
        }

    return {
        "idx": idx,
        "params": params,
        "ok": True,
        "rc": 0,
        "osmium": float(product_match.group(4)),
        "total": float(total_match.group(1)),
    }


def main():
    results = []
    for i, params in enumerate(CANDIDATES, start=1):
        result = run_candidate(i, params)
        results.append(result)
        if result["ok"]:
            print(f"#{i:02d} osmium={result['osmium']:.2f} total={result['total']:.2f} params={result['params']}")
        else:
            print(f"#{i:02d} FAILED rc={result['rc']} params={result['params']}")

    ok_results = [r for r in results if r["ok"]]
    ok_results.sort(key=lambda r: (r["osmium"], r["total"]), reverse=True)

    print("\nTOP 5 (by Osmium, then Total):")
    for r in ok_results[:5]:
        print(f"idx={r['idx']} osmium={r['osmium']:.2f} total={r['total']:.2f} params={r['params']}")


if __name__ == "__main__":
    main()
