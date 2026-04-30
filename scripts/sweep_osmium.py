import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
base = (root / "traders" / "round2_trader.py").read_text()
out = root / "traders" / "sweep_tmp"
out.mkdir(exist_ok=True)

configs = [
    (24, 0.28, 0.10, 0.8, 22, 18, 5, 18),
    (24, 0.30, 0.12, 0.7, 24, 20, 4, 16),
    (30, 0.26, 0.10, 0.8, 22, 18, 5, 20),
    (20, 0.32, 0.14, 0.6, 24, 22, 4, 14),
    (36, 0.22, 0.08, 1.0, 18, 14, 6, 24),
    (28, 0.30, 0.10, 0.7, 20, 18, 5, 20),
]

pattern = re.compile(r"ASH_COATED_OSMIUM\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)")

for i, (window, fast, slow, band, take, post, short_w, long_w) in enumerate(configs, 1):
    text = base
    replacements = [
        ("WINDOW = 30", f"WINDOW = {window}"),
        ("SHORT_WINDOW = 6", f"SHORT_WINDOW = {short_w}"),
        ("LONG_WINDOW = 24", f"LONG_WINDOW = {long_w}"),
        ("FAST_ALPHA = 0.22", f"FAST_ALPHA = {fast}"),
        ("SLOW_ALPHA = 0.08", f"SLOW_ALPHA = {slow}"),
        ("ENTRY_BAND = 1.0", f"ENTRY_BAND = {band}"),
        ("MAX_TAKE_PER_LEVEL = 18", f"MAX_TAKE_PER_LEVEL = {take}"),
        ("MAX_POST_SIZE = 14", f"MAX_POST_SIZE = {post}"),
    ]

    ok = True
    for old, new in replacements:
        if old not in text:
            ok = False
            break
        text = text.replace(old, new, 1)

    if not ok:
        print(f"cfg {i}: pattern mismatch")
        continue

    trader_path = out / f"cfg_{i}.py"
    trader_path.write_text(text)

    cmd = ["make", "round2", f"TRADER={trader_path}"]
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    output = result.stdout + "\n" + result.stderr

    match = pattern.search(output)
    if match:
        print(
            f"cfg {i}: params={(window, fast, slow, band, take, post, short_w, long_w)} osmium_total={match.group(4)}"
        )
    else:
        print(f"cfg {i}: parse failed rc={result.returncode}")
