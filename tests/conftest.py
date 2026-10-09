"""Shared fixtures. Tests use synthetic data only: no API keys, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from printtime.config import Settings, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config(tmp_path: Path, extra: list[str] | None = None) -> Settings:
    over = [
        f"paths.raw={tmp_path}/raw",
        f"paths.processed={tmp_path}/processed",
        f"paths.calendar={tmp_path}/calendar",
        f"paths.reports={tmp_path}/reports",
        f"paths.figures={tmp_path}/reports/figures",
        f"paths.tables={tmp_path}/reports/tables",
        f"paths.data_quality={tmp_path}/reports/data_quality",
        f"databento.spend_ledger={tmp_path}/raw/spend_ledger.json",
        "synthetic.window_s=[-300, 600]",
    ]
    return load_config(ROOT / "config.yaml", over + (extra or []))


test_config.__test__ = False  # type: ignore[attr-defined]


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    return test_config(tmp_path)


@pytest.fixture(scope="session")
def base_cfg(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return test_config(tmp_path_factory.mktemp("base"))
