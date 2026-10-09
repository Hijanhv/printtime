"""Typed configuration loaded from config.yaml.

Pydantic validates the whole file at load time, so a typo, a wrong type or an
impossible value (a negative budget, a window that ends before it starts)
fails immediately with a clear message instead of deep inside a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _window(v: tuple[int, int]) -> tuple[int, int]:
    if v[0] >= v[1]:
        raise ValueError(f"window start must be before end, got {v}")
    return v


class Paths(_Strict):
    raw: Path
    processed: Path
    calendar: Path
    reports: Path
    figures: Path
    tables: Path
    data_quality: Path


class Sample(_Strict):
    start: str
    end: str

    @model_validator(mode="after")
    def _ordered(self) -> Sample:
        if self.start >= self.end:
            raise ValueError("sample.start must be before sample.end")
        return self


class Instrument(_Strict):
    root: str
    name: str
    venue: Literal["CBOT", "CME", "COMEX", "NYMEX"]
    mbp10: bool

    @property
    def databento_symbol(self) -> str:
        """Front contract by volume, Databento continuous symbology."""
        return f"{self.root}.v.0"


class EventType(_Strict):
    tier: Literal[1, 2]
    times_et: list[str] = Field(min_length=1)
    fred_release_search: str | None

    @field_validator("times_et")
    @classmethod
    def _hhmm(cls, v: list[str]) -> list[str]:
        for t in v:
            h, _, m = t.partition(":")
            if not (h.isdigit() and m.isdigit() and 0 <= int(h) < 24 and 0 <= int(m) < 60):
                raise ValueError(f"time must be HH:MM, got {t!r}")
        return v


class Windows(_Strict):
    ohlcv_1s: tuple[int, int]
    mbp_1: tuple[int, int]
    mbp_10: tuple[int, int]

    _check = field_validator("ohlcv_1s", "mbp_1", "mbp_10")(_window)


class Databento(_Strict):
    dataset: str
    stype_in: str
    budget_usd: float = Field(gt=0)
    spend_ledger: Path
    windows: Windows


class Fred(_Strict):
    base_url: str
    api_key_env: str
    first_release_series: dict[str, str]


class Grid(_Strict):
    step_ms: int = Field(gt=0)
    start_s: int
    end_s: int

    @model_validator(mode="after")
    def _ordered(self) -> Grid:
        if self.start_s >= self.end_s:
            raise ValueError("grid start_s must be before end_s")
        return self


class Panel(_Strict):
    fine_grid: Grid
    coarse_grid: Grid
    baseline_window_s: tuple[int, int]
    depth_levels: list[int]

    _check = field_validator("baseline_window_s")(_window)


class ControlDays(_Strict):
    per_time: dict[str, int]
    match_weekday: bool


class Execution(_Strict):
    order_sizes: list[int] = Field(min_length=1)
    fee_per_contract_side_usd: float = Field(ge=0)
    normal_cost_multiple: float = Field(gt=1)


class Strategy(_Strict):
    delays_s: list[int] = Field(min_length=1)
    exits_s: list[int] = Field(min_length=1)
    min_train_events: int = Field(ge=1)


class Analysis(_Strict):
    reaction_horizons_s: list[int] = Field(min_length=1)
    first_move_quantile: float = Field(gt=0, lt=1)
    lead_lag_window_s: int = Field(gt=0)
    recovery_levels: list[float]
    bootstrap_reps: int = Field(gt=0)
    fdr_alpha: float = Field(gt=0, lt=1)
    fomc_surprise_window_s: tuple[int, int]
    execution: Execution
    strategy: Strategy

    _check = field_validator("fomc_surprise_window_s")(_window)


class Monitoring(_Strict):
    metrics_port: int
    alert_spread_ticks: int = Field(gt=0)
    alert_depth_pct_of_baseline: float = Field(gt=0, le=100)
    alert_no_data_s: float = Field(gt=0)


class SyntheticEvent(_Strict):
    withdrawal_lead_s: float = Field(ge=0)
    depth_drop: float = Field(ge=0, lt=1)
    depth_recovery_half_life_s: float = Field(gt=0)
    spread_blowout_ticks: int = Field(ge=0)
    spread_half_life_s: float = Field(gt=0)
    activity_multiplier: float = Field(ge=1)
    activity_half_life_s: float = Field(gt=0)


class SyntheticInstrument(_Strict):
    tick_size: float = Field(gt=0)
    start_price: float = Field(gt=0)
    base_depth: int = Field(gt=0)
    update_rate: float = Field(gt=0)
    trade_share: float = Field(ge=0, lt=1)
    vol_ticks_per_s: float = Field(ge=0)
    jump_ticks_per_sd: float
    lag_ms: int = Field(ge=0)


class Synthetic(_Strict):
    seed: int
    window_s: tuple[int, int]
    levels: int = Field(ge=1)
    event: SyntheticEvent
    instruments: dict[str, SyntheticInstrument]

    _check = field_validator("window_s")(_window)


class Settings(_Strict):
    paths: Paths
    timezone: str
    sample: Sample
    instruments: dict[str, Instrument]
    optional_instruments: dict[str, Instrument]
    events: dict[str, EventType]
    databento: Databento
    fred: Fred
    panel: Panel
    control_days: ControlDays
    analysis: Analysis
    monitoring: Monitoring
    synthetic: Synthetic

    @model_validator(mode="after")
    def _mbp10_subset(self) -> Settings:
        if not any(i.mbp10 for i in self.instruments.values()):
            raise ValueError("at least one instrument needs mbp10: true for execution-cost curves")
        return self

    def tier(self, tier: int) -> list[str]:
        return [name for name, e in self.events.items() if e.tier == tier]


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Settings:
    """Load and validate config.yaml.

    overrides are "dotted.key=value" strings; each value is parsed as YAML.
    """
    raw = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(value))
    return Settings.model_validate(raw)
