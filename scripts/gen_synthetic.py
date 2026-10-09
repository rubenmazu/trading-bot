"""Generează un set de date sintetic pentru demo-ul de backtest (data/synthetic.csv).

Folosește generatorul determinist din tests/fixtures/synthetic.py. Instrumentul și intervalul
corespund config/backtest.toml (XYZ, bare de 15 minute). Rulează:

    uv run python scripts/gen_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fixtures.synthetic import SyntheticSpec, generate_bars, write_dataset  # noqa: E402

OUT = ROOT / "data" / "synthetic.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spec = SyntheticSpec(
        scenario="mean_reverting",
        seed=42,
        n_bars=600,
        instrument="XYZ",
        interval_min=15,
    )
    bars = generate_bars(spec)
    manifest = write_dataset(OUT, bars, source_id="synthetic")
    print(f"Scris {len(bars)} bare în {OUT}")
    print(f"Interval: {manifest.start} .. {manifest.end}")
    print(f"source_id: {manifest.source_id}")


if __name__ == "__main__":
    main()
