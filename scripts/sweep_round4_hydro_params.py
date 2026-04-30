from pathlib import Path
import os
import random
import re
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = PROJECT_ROOT / "traders/round4_trader2.py"
TMP_FILE = PROJECT_ROOT / "traders/_tmp_hydro_sweep.py"

param_grid = {
    "HYDRO_EMA_ALPHA": [0.10, 0.12, 0.15, 0.18],
    "HYDRO_TAKE_EDGE": [1, 2, 3],
    "HYDRO_SKEW": [0.03, 0.04, 0.05],
    "HYDRO_HEAVY_POS": [140, 150, 170],
    "HYDRO_SOFT_LONG": [150, 160, 180],
    "HYDRO_SOFT_SHORT": [-160, -140, -120],
    "HYDRO_MAX_POST": [20, 30, 40],
    "HYDRO_HALF_SPREAD": [3, 4, 5],
}


def build_trials(count: int = 24):
    random.seed(11)
    trials = []
    for _ in range(count):
        trials.append({k: random.choice(v) for k, v in param_grid.items()})

    trials.append(
        {
            "HYDRO_EMA_ALPHA": 0.15,
            "HYDRO_TAKE_EDGE": 2,
            "HYDRO_SKEW": 0.04,
            "HYDRO_HEAVY_POS": 150,
            "HYDRO_SOFT_LONG": 160,
            "HYDRO_SOFT_SHORT": -120,
            "HYDRO_MAX_POST": 30,
            "HYDRO_HALF_SPREAD": 4,
        }
    )

    seen = set()
    uniq = []
    for trial in trials:
        key = tuple(sorted(trial.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(trial)
    return uniq


def patch_text(base_text: str, params: dict) -> str:
    text = base_text
    for key, value in params.items():
        text = re.sub(rf"^{key}\\s*=.*$", f"{key}   = {value}", text, flags=re.MULTILINE)
    return text


def parse_hydro(stdout: str):
    hydro = None
    for line in stdout.splitlines():
        if line.strip().startswith("HYDROGEL_PACK"):
            try:
                hydro = float(line.split()[-1])
            except Exception:
                hydro = None
    return hydro


def main():
    os.chdir(PROJECT_ROOT)
    base = BASE_FILE.read_text()
    results = []
    trials = build_trials()

    for i, params in enumerate(trials, start=1):
        TMP_FILE.write_text(patch_text(base, params))

        proc = subprocess.run(
            ["make", "-C", str(Path.cwd()), "round4", f"TRADER={TMP_FILE}"],
            capture_output=True,
            text=True,
        )
        out = proc.stdout + "\n" + proc.stderr
        hydro = parse_hydro(out)

        if hydro is None:
            print(f"run {i:02d}/{len(trials)} failed")
            continue

        results.append((hydro, params))
        print(f"run {i:02d}/{len(trials)} hydro={hydro:.2f}")

    TMP_FILE.unlink(missing_ok=True)

    results.sort(key=lambda x: x[0], reverse=True)
    print("\nTOP CANDIDATES")
    for hydro, params in results[:10]:
        print(hydro, params)


if __name__ == "__main__":
    main()
