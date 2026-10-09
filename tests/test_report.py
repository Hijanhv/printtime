"""Phase 9: the report is assembled from tables and keeps synthetic runs separate."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from printtime.config import Settings
from printtime.report import README_END, README_START, build_report, update_readme


def regression_table() -> pl.DataFrame:
    rows = []
    for inst, beta, q, n, surv in (
        ("ZN", -4.0, 0.001, 20, True),
        ("ES", -1.0, 0.30, 20, False),
        ("GC", -2.0, 0.01, 10, True),
    ):
        rows.append(
            {
                "event_type": "CPI",
                "instrument": inst,
                "horizon_s": 60,
                "unit": "ret_ticks",
                "beta": beta,
                "boot_lo": beta - 1,
                "boot_hi": beta + 1,
                "r2": 0.5,
                "n_events": n,
                "q_bh": q,
                "survives_fdr": surv,
                "mechanical": False,
            }
        )
    return pl.DataFrame(rows)


def setup(cfg: Settings, synthetic: bool) -> None:
    t = cfg.paths.tables
    t.mkdir(parents=True, exist_ok=True)
    regression_table().write_parquet(t / "reaction_regressions.parquet")
    (t / "liquidity_summary.md").write_text(
        "table\n\n- ZN, CPI release: top-of-book depth falls by 80%\n"
    )
    cfg.paths.calendar.mkdir(parents=True, exist_ok=True)
    if synthetic:
        pl.DataFrame({"window_id": ["x"]}).write_parquet(
            cfg.paths.calendar / "synthetic_scenarios.parquet"
        )


def test_synthetic_report_has_a_banner_and_never_touches_the_readme(
    cfg: Settings, tmp_path: Path
) -> None:
    setup(cfg, synthetic=True)
    text = build_report(cfg).read_text()
    assert "SYNTHETIC: NOT RESULTS" in text
    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{README_START}\nold\n{README_END}\n")
    assert not update_readme(cfg, readme) and "old" in readme.read_text()


def test_real_report_labels_robust_and_suggestive(cfg: Settings, tmp_path: Path) -> None:
    setup(cfg, synthetic=False)
    text = build_report(cfg).read_text()
    assert "SYNTHETIC" not in text
    assert (
        "Of 3 reaction-function tests, 1 are **robust**" in text
    )  # GC survives FDR but has only 10 events
    assert (
        "| CPI | ZN | 60 s | -4.00 |" in text and "| robust |" in text and "| suggestive |" in text
    )
    assert "_Not run yet._" in text  # missing analyses are said to be missing, not invented
    assert "top-of-book depth falls by 80%" in text
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\nend\n")
    assert update_readme(cfg, readme)
    new = readme.read_text()
    assert "old" not in new and "depth falls by 80%" in new and new.endswith("end\n")
