"""Phase 1: FRED calendar, FOMC file, surprises validator, standardisation, ALFRED check.

FRED is replaced by an in-memory fake server (httpx.MockTransport): no network,
no key. Release names and IDs in the fake are made up for the test.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from printtime.calendar import fred as fred_mod
from printtime.calendar.alfred import cross_check, mismatches, published_value
from printtime.calendar.events import (
    build_calendar,
    fomc_event_rows,
    fred_event_rows,
    validate_fomc,
    write_fomc_template,
)
from printtime.calendar.fred import FredClient, FredError
from printtime.calendar.surprises import (
    SURPRISE_COLUMNS,
    standardise,
    summary,
    validate_surprises,
    write_surprises_template,
)
from printtime.cli import app
from printtime.config import Settings
from tests.conftest import ROOT

RELEASES = [
    {"id": 101, "name": "Consumer Price Index"},
    {"id": 102, "name": "Consumer Price Index for All Urban Consumers: Research Series"},
    {"id": 201, "name": "Employment Situation"},
    {"id": 301, "name": "Producer Price Index"},
]
DATES = {
    101: ["2024-09-11", "2024-10-10", "2024-11-13", "2026-10-15"],
    201: ["2024-10-04", "2024-11-01", "2024-12-06"],
}


def fake_fred(seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.params.get("api_key") != "test-key":
            return httpx.Response(400, json={"error_message": "Bad Request. api_key not set."})
        path = request.url.path.rstrip("/").split("/")[-1]
        if path == "releases":
            offset = int(request.url.params.get("offset", 0))
            page = RELEASES[offset : offset + 2]  # tiny pages exercise pagination
            return httpx.Response(200, json={"count": len(RELEASES), "releases": page})
        if path == "dates":
            rid = int(request.url.params["release_id"])
            return httpx.Response(
                200, json={"release_dates": [{"release_id": rid, "date": d} for d in DATES[rid]]}
            )
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler)


def client(seen: list[httpx.Request] | None = None) -> FredClient:
    return FredClient("test-key", "https://fred.test/fred", transport=fake_fred(seen))


# --- FRED --------------------------------------------------------------------------------


def test_release_ids_are_looked_up_by_name_not_hard_coded() -> None:
    seen: list[httpx.Request] = []
    c = client(seen)
    assert c.find_release_id("Consumer Price Index") == 101  # exact match wins over the longer name
    assert c.find_release_id("employment situation") == 201
    assert sum(1 for r in seen if r.url.path.endswith("releases")) >= 2  # paginated


def test_ambiguous_or_missing_names_fail_loudly() -> None:
    with pytest.raises(FredError, match="matched 3"):
        client().find_release_id("Price Index")  # no exact match, three partial matches
    with pytest.raises(FredError, match="matched 0"):
        client().find_release_id("Beige Book")


def test_release_dates_are_clipped_to_the_sample() -> None:
    dates = client().release_dates(101, dt.date(2024, 10, 1), dt.date(2026, 9, 30))
    assert dates == [dt.date(2024, 10, 10), dt.date(2024, 11, 13)]


def test_http_errors_do_not_leak_the_key() -> None:
    bad = FredClient("wrong", "https://fred.test/fred", transport=fake_fred())
    with pytest.raises(FredError) as err:
        bad.releases()
    assert "wrong" not in str(err.value) and "api_key=" not in str(err.value)


# --- calendar and FOMC ------------------------------------------------------------------


def test_calendar_rows_have_dst_correct_t0(cfg: Settings) -> None:
    rows = fred_event_rows(cfg, client(), ["CPI", "NFP"])
    cal = build_calendar(rows)
    assert cal.height == 5
    cpi = cal.filter(pl.col("event_id") == "CPI_2024-11-13").row(0, named=True)
    assert cpi["t0_utc"].startswith("2024-11-13T13:30")  # 08:30 EST after the November change
    nfp = cal.filter(pl.col("event_id") == "NFP_2024-10-04").row(0, named=True)
    assert nfp["t0_utc"].startswith("2024-10-04T12:30")  # 08:30 EDT
    assert (np.diff(cal["t0_utc_ns"].to_numpy()) >= 0).all()


def test_duplicate_events_are_rejected(cfg: Settings) -> None:
    rows = fred_event_rows(cfg, client(), ["CPI"])
    with pytest.raises(ValueError, match="duplicate"):
        build_calendar(rows + rows[:1])


def test_fomc_template_is_created_once_and_validated(cfg: Settings, tmp_path: Path) -> None:
    path = tmp_path / "fomc_dates.csv"
    assert write_fomc_template(path)
    assert not write_fomc_template(path)  # never overwrites your file
    _, rep = validate_fomc(cfg, path)
    assert not rep.ok and "no FOMC meetings" in rep.errors[0]
    path.write_text(
        "date,statement_time_et,press_conference_time_et,source_url\n"
        "2024-11-07,14:00,14:30,https://www.federalreserve.gov/x\n"
        "2024-11-07,14:00,14:30,https://www.federalreserve.gov/x\n"  # duplicate
        "2024-12-21,14:00,14:30,https://www.federalreserve.gov/x\n"  # Saturday
        "2025-01-29,2pm,14:30,https://www.federalreserve.gov/x\n"  # bad time
        "2025-03-19,14:00,13:30,https://www.federalreserve.gov/x\n"  # press before statement
        "2025-05-07,14:00,,not-a-link\n"  # no URL
    )
    _, rep = validate_fomc(cfg, path)
    text = " | ".join(rep.errors)
    for needle in ("duplicate", "weekend", "not HH:MM", "not after the statement", "source_url"):
        assert needle in text


def test_fomc_rows_carry_statement_and_press_conference(cfg: Settings, tmp_path: Path) -> None:
    path = tmp_path / "fomc_dates.csv"
    path.write_text(
        "date,statement_time_et,press_conference_time_et,source_url\n"
        "2025-06-18,14:00,14:30,https://www.federalreserve.gov/a\n"
        "2023-06-14,14:00,14:30,https://www.federalreserve.gov/b\n"  # before the sample: ignored
    )
    df, rep = validate_fomc(cfg, path)
    assert rep.ok and rep.warnings
    cal = build_calendar(fomc_event_rows(cfg, df))
    assert cal["stage"].to_list() == ["statement", "press_conference"]
    assert cal["t0_utc"].to_list()[0].startswith("2025-06-18T18:00")


# --- surprises ----------------------------------------------------------------------------

HEADER = ",".join(SURPRISE_COLUMNS) + "\n"
URL = "https://news.example/cpi"


def write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "surprises.csv"
    p.write_text(HEADER + "".join(line + "\n" for line in lines))
    return p


def test_surprises_template_and_empty_file(cfg: Settings, tmp_path: Path) -> None:
    p = tmp_path / "surprises.csv"
    assert write_surprises_template(p) and not write_surprises_template(p)
    _, rep = validate_surprises(cfg, p)
    assert not rep.ok and "no rows" in rep.errors[0]


def test_validator_catches_hand_entry_mistakes(cfg: Settings, tmp_path: Path) -> None:
    p = write(
        tmp_path,
        [
            f"CPI_2024-10-10,CPI,2024-10-10,core_cpi_mom,0.3,0.2,0.3,pct,{URL},",
            f"CPI_2024-10-10,CPI,2024-10-10,core_cpi_mom,0.3,0.2,0.3,pct,{URL},",  # duplicate
            f"CPI_2024-11-13,CPI,2024-11-13,core_cpi_mom,,0.3,0.3,pct,{URL},",  # missing actual
            f"CPI_2024-12-11,CPI,2024-12-11,core_cpi_mom,0.3,0.3,0.3,percent,{URL},",  # bad unit
            # not a number
            f"NFP_2024-11-01,NFP,2024-11-01,nfp_change,12k,113,223,thousands,{URL},",
            # wrong variable for type
            f"NFP_2024-12-06,NFP,2024-12-06,core_cpi_mom,0.3,0.3,0.3,pct,{URL},",
            # FOMC has no consensus
            f"FOMC_2024-11-07,FOMC,2024-11-07,core_cpi_mom,0.3,0.3,0.3,pct,{URL},",
            # event_id mismatch
            f"CPI_2024-10-11,CPI,2024-10-10,core_cpi_mom,0.3,0.2,0.3,pct,{URL},",
            "CPI_2025-01-15,CPI,2025-01-15,core_cpi_mom,0.2,0.2,0.3,pct,,",  # no source
            # implausible: typo?
            f"CPI_2025-02-12,CPI,2025-02-12,core_cpi_mom,3.0,0.3,0.2,pct,{URL},",
        ],
    )
    _, rep = validate_surprises(cfg, p)
    errors = " | ".join(rep.errors)
    for needle in (
        "duplicate",
        "actual is missing",
        "unit for core_cpi_mom",
        "not a number",
        "belongs to CPI",
        "must be CPI or NFP",
        "should be 'CPI_2024-10-10'",
        "source_url",
    ):
        assert needle in errors, needle
    assert any("plausible range" in w for w in rep.warnings)


def test_validator_checks_dates_against_the_calendar(cfg: Settings, tmp_path: Path) -> None:
    cal = build_calendar(fred_event_rows(cfg, client(), ["CPI", "NFP"]))
    p = write(
        tmp_path,
        [
            f"CPI_2024-10-10,CPI,2024-10-10,core_cpi_mom,0.3,0.2,0.3,pct,{URL},",
            f"CPI_2024-10-16,CPI,2024-10-16,core_cpi_mom,0.3,0.2,0.3,pct,{URL},",  # no such release
        ],
    )
    _, rep = validate_surprises(cfg, p, cal)
    assert any("CPI_2024-10-16 is not in releases.csv" in e for e in rep.errors)
    assert any("have no core_cpi_mom row yet" in w for w in rep.warnings)


@settings(max_examples=60, deadline=None)
@given(
    actual=st.floats(min_value=-1.0, max_value=1.5, allow_nan=False).map(lambda x: round(x, 1)),
    consensus=st.floats(min_value=-1.0, max_value=1.5, allow_nan=False).map(lambda x: round(x, 1)),
    unit=st.sampled_from(["pct", "thousands", "", "%"]),
)
def test_valid_rows_pass_and_wrong_units_never_do(
    base_cfg: Settings,
    tmp_path_factory: pytest.TempPathFactory,
    actual: float,
    consensus: float,
    unit: str,
) -> None:
    p = write(
        tmp_path_factory.mktemp("s"),
        [
            f"CPI_2024-10-10,CPI,2024-10-10,core_cpi_mom,{actual},{consensus},0.2,{unit},{URL},",
        ],
    )
    df, rep = validate_surprises(base_cfg, p)
    assert rep.ok == (unit == "pct")
    if rep.ok:
        assert df["actual"][0] == actual and df["consensus"][0] == consensus


def test_standardised_surprise_by_hand() -> None:
    df = pl.DataFrame(
        {
            "event_type": ["CPI"] * 4,
            "variable": ["core_cpi_mom"] * 4,
            "release_date": [dt.date(2025, m, 12) for m in (1, 2, 3, 4)],
            "actual": [0.3, 0.4, 0.2, 0.3],
            "consensus": [0.2, 0.3, 0.3, 0.3],
        }
    )
    z = standardise(df)
    s = np.array([0.1, 0.1, -0.1, 0.0])
    np.testing.assert_allclose(z["surprise"].to_numpy(), s, atol=1e-12)
    np.testing.assert_allclose(z["z"].to_numpy(), s / np.std(s, ddof=1), atol=1e-12)
    zx = standardise(df, expanding=True)["z"].to_numpy()
    assert np.isnan(zx[:2]).all()  # needs two earlier surprises
    # The first two surprises are both 0.1 (after removing float noise), so
    # their spread is zero and the third z is undefined rather than huge.
    assert np.isnan(zx[2])
    assert zx[3] == pytest.approx(0.0 / np.std(s[:3], ddof=1))
    assert summary(df)["events"][0] == 4


def test_expanding_scale_never_uses_the_current_or_future_events() -> None:
    rng = np.random.default_rng(0)
    n = 12
    base = pl.DataFrame(
        {
            "event_type": ["NFP"] * n,
            "variable": ["nfp_change"] * n,
            "release_date": [dt.date(2025, 1, 1) + dt.timedelta(days=30 * k) for k in range(n)],
            "actual": rng.normal(150, 60, n).round(),
            "consensus": rng.normal(150, 20, n).round(),
        }
    )
    z0 = standardise(base, expanding=True)["z"].to_numpy()
    k = 7
    changed = base.with_columns(
        pl.when(pl.int_range(n) >= k).then(pl.col("actual") + 999).otherwise(pl.col("actual"))
    )
    z1 = standardise(changed, expanding=True)["z"].to_numpy()
    np.testing.assert_array_equal(
        z0[:k], z1[:k]
    )  # changing later events leaves earlier z untouched


# --- ALFRED -------------------------------------------------------------------------------


def vintages() -> pl.DataFrame:
    """Hand-built vintages: the Nov release revises October, as real releases do."""
    d = dt.date
    return pl.DataFrame(
        {
            "realtime_start": [d(2024, 10, 10), d(2024, 10, 10), d(2024, 11, 13), d(2024, 11, 13)],
            "date": [d(2024, 8, 1), d(2024, 9, 1), d(2024, 9, 1), d(2024, 10, 1)],
            "value": [100.0, 100.3, 100.25, 100.55],
        }
    )


def test_published_value_uses_the_vintage_of_the_release_day() -> None:
    month, v = published_value(vintages(), dt.date(2024, 11, 13), "pct_change")  # type: ignore[misc]
    assert month == dt.date(2024, 10, 1)
    assert v == pytest.approx((100.55 / 100.25 - 1) * 100)  # uses revised September, not 100.3
    _, diff = published_value(vintages(), dt.date(2024, 11, 13), "diff")  # type: ignore[misc]
    assert diff == pytest.approx(0.30)
    _, lvl = published_value(vintages(), dt.date(2024, 11, 13), "level")  # type: ignore[misc]
    assert lvl == 100.55
    assert published_value(vintages(), dt.date(2024, 11, 14), "level") is None


def test_cross_check_flags_mismatches_beyond_rounding(cfg: Settings) -> None:
    s = pl.DataFrame(
        {
            "event_id": ["CPI_2024-11-13", "CPI_2024-10-10", "CPI_2024-12-11"],
            "variable": ["core_cpi_mom"] * 3,
            "release_date": [dt.date(2024, 11, 13), dt.date(2024, 10, 10), dt.date(2024, 12, 11)],
            "actual": [0.3, 0.5, 0.3],
        }
    )
    check = cross_check(cfg, s, lambda series: vintages())
    status = dict(zip(check["event_id"], check["status"], strict=True))
    assert status["CPI_2024-11-13"] == "ok"  # 0.299% rounds to 0.3
    assert status["CPI_2024-10-10"] == "MISMATCH"  # published 0.3, typed 0.5
    assert status["CPI_2024-12-11"].startswith("no ALFRED vintage")
    assert mismatches(check).height == 2


# --- CLI ----------------------------------------------------------------------------------


def test_calendar_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(
        fred_mod,
        "FredClient",
        lambda key, base, timeout: FredClient(key, base, timeout, transport=fake_fred()),
    )
    base = [
        "--config",
        str(ROOT / "config.yaml"),
        "--set",
        f"paths.calendar={tmp_path}",
        "--set",
        f"paths.tables={tmp_path}/tables",
    ]
    r = CliRunner().invoke(app, [*base, "calendar", "build"])
    assert r.exit_code == 0, r.output
    assert "created template" in r.output and "FOMC dates not included yet" in r.output
    assert (tmp_path / "releases.csv").exists()
    (tmp_path / "surprises.csv").write_text(
        HEADER + f"CPI_2024-10-10,CPI,2024-10-10,core_cpi_mom,0.3,0.2,0.3,pct,{URL},\n"
    )
    r = CliRunner().invoke(app, [*base, "calendar", "validate", "--skip-alfred"])
    assert r.exit_code == 1  # FOMC file is still empty: validation must fail loudly
    assert "no FOMC meetings" in r.output
    (tmp_path / "fomc_dates.csv").write_text(
        "date,statement_time_et,press_conference_time_et,source_url\n2024-11-07,14:00,14:30,https://www.federalreserve.gov/x\n"
    )
    r = CliRunner().invoke(app, [*base, "calendar", "build"])
    assert r.exit_code == 0 and "FOMC   press_conference  1" in r.output, r.output
    r = CliRunner().invoke(app, [*base, "calendar", "validate", "--skip-alfred"])
    assert r.exit_code == 0, r.output
    assert "calendar validate: passed" in r.output and "core_cpi_mom" in r.output
