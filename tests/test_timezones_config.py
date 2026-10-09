"""Release-time conversion across DST changes, and config validation."""

from __future__ import annotations

import datetime as dt
import re

import pydantic
import pytest

from printtime.calendar.timezones import et_to_utc, et_to_utc_ns, utc_ns_to_et
from printtime.config import load_config
from tests.conftest import ROOT


@pytest.mark.parametrize(
    ("date", "hhmm", "utc"),
    [
        # 2025 spring change: Sunday 9 March. CPI-style 08:30 releases either side.
        (dt.date(2025, 3, 7), "08:30", "13:30"),  # Friday before: EST, UTC-5
        (dt.date(2025, 3, 10), "08:30", "12:30"),  # Monday after: EDT, UTC-4
        # 2025 autumn change: Sunday 2 November. FOMC-style 14:00 either side.
        (dt.date(2025, 10, 29), "14:00", "18:00"),  # EDT
        (dt.date(2025, 11, 5), "14:00", "19:00"),  # EST
    ],
)
def test_release_times_follow_daylight_saving(date: dt.date, hhmm: str, utc: str) -> None:
    assert et_to_utc(date, hhmm).strftime("%H:%M") == utc
    assert et_to_utc(date, hhmm).date() == date


def test_ns_round_trip_is_exact() -> None:
    ns = et_to_utc_ns(dt.date(2026, 3, 11), "08:30")
    assert ns % 1_000_000_000 == 0
    back = utc_ns_to_et(ns)
    assert (back.hour, back.minute, back.date()) == (8, 30, dt.date(2026, 3, 11))


def test_config_loads_and_derives_symbols() -> None:
    c = load_config(ROOT / "config.yaml")
    assert c.instruments["ZN"].databento_symbol == "ZN.v.0"
    assert {k for k, v in c.instruments.items() if v.mbp10} == {"ZT", "ZN", "ES"}
    assert set(c.tier(1)) == {"CPI", "NFP", "FOMC"}
    assert c.databento.budget_usd == 40.0


@pytest.mark.parametrize(
    "override",
    [
        "databento.budget_usd=-5",
        "databento.windows.mbp_1=[60,-60]",
        "events.CPI.times_et=['8:3x']",
        "instruments.ZN.venue=MOON",
        "panel.fine_grid.step_ms=0",
        "unknown_key=1",
    ],
)
def test_config_rejects_bad_values(override: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        load_config(ROOT / "config.yaml", [override])


def test_no_crypto_anywhere_in_config() -> None:
    # Scope guard (spec, top): whole words only, so "definition" does not match "defi".
    text = (ROOT / "config.yaml").read_text().lower()
    banned = r"\b(crypto\w*|bitcoin|btc|eth|ethereum|web3|defi|tokens?|blockchain|ccxt)\b"
    assert re.search(banned, text) is None
