"""Budgeted, cached Databento downloads and conversion to the canonical book frame.

Three guards stand between the code and the credit:

1. **Cache first.** Each (schema, instrument, window) is one .dbn.zst file in
   data/raw/. If it exists, nothing is requested.
2. **Quote before buying.** `metadata.get_cost` (a free metadata call) is
   logged before every download.
3. **Spend ledger.** Every purchase is appended to data/raw/spend_ledger.json;
   a request that would take cumulative spend past `databento.budget_usd`
   raises BudgetExceededError. Nothing is bought without confirm=True.

The ledger is shared knowledge with the other project only through the
budget number: set `databento.budget_usd` to what is left for Print Time.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from printtime.config import Settings
from printtime.log import get_logger
from printtime.schema import ADD, ASK, BID, CANCEL, CLEAR, FILL, MODIFY, NO_PRICE, NONE, TRADE
from printtime.secrets import require_key

log = get_logger(__name__)

FIXED_PRICE_SCALE = 1e-9  # Databento prices are int64 in units of 1e-9
UNDEF_PRICE = np.iinfo(np.int64).max
ACTION_MAP = {"A": ADD, "C": CANCEL, "M": MODIFY, "T": TRADE, "F": FILL, "R": CLEAR, "N": MODIFY}
SIDE_MAP = {"B": BID, "A": ASK, "N": NONE}
NS = 1_000_000_000


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class Request:
    """One window of one schema for one instrument."""

    instrument: str
    schema: str
    window_id: str  # event_id plus stage, or a control-day id
    start_ns: int
    end_ns: int
    symbol: str

    def path(self, raw_root: Path) -> Path:
        return raw_root / self.schema / self.instrument / f"{self.window_id}.dbn.zst"

    @property
    def start_iso(self) -> str:
        return dt.datetime.fromtimestamp(self.start_ns / NS, dt.UTC).isoformat()

    @property
    def end_iso(self) -> str:
        return dt.datetime.fromtimestamp(self.end_ns / NS, dt.UTC).isoformat()


def make_request(
    cfg: Settings,
    instrument: str,
    schema: str,
    window_id: str,
    t0_ns: int,
    window_s: tuple[int, int],
) -> Request:
    spec = cfg.instruments.get(instrument) or cfg.optional_instruments[instrument]
    return Request(
        instrument,
        schema,
        window_id,
        t0_ns + window_s[0] * NS,
        t0_ns + window_s[1] * NS,
        spec.databento_symbol,
    )


class SpendLedger:
    """Append-only record of purchases, kept next to the raw data."""

    def __init__(self, path: Path, budget_usd: float) -> None:
        self.path = path
        self.budget = budget_usd

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data: list[dict[str, Any]] = json.loads(self.path.read_text())
        return data

    @property
    def spent(self) -> float:
        return float(sum(e["cost_usd"] for e in self.entries()))

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def check(self, cost: float) -> None:
        if self.spent + cost > self.budget + 1e-9:
            raise BudgetExceededError(
                f"request costs ${cost:.4f}; already spent ${self.spent:.4f} "
                f"of the ${self.budget:.2f} budget"
            )

    def record(self, request: Request, cost: float) -> None:
        entries = self.entries()
        entries.append(
            {
                "when": dt.datetime.now(dt.UTC).isoformat(),
                "instrument": request.instrument,
                "schema": request.schema,
                "window_id": request.window_id,
                "cost_usd": cost,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2))


def client(cfg: Settings) -> Any:
    import databento as db

    return db.Historical(require_key("DATABENTO_API_KEY"))


def _params(cfg: Settings, r: Request) -> dict[str, Any]:
    return {
        "dataset": cfg.databento.dataset,
        "symbols": [r.symbol],
        "schema": r.schema,
        "start": r.start_iso,
        "end": r.end_iso,
        "stype_in": cfg.databento.stype_in,
    }


def quote(cfg: Settings, r: Request, api: Any) -> float:
    """Price of a request in USD. A free metadata call: nothing is bought."""
    return float(api.metadata.get_cost(**_params(cfg, r)))


def billable_bytes(cfg: Settings, r: Request, api: Any) -> int:
    """Uncompressed size Databento would bill for. Also a free metadata call."""
    return int(api.metadata.get_billable_size(**_params(cfg, r)))


def download(cfg: Settings, r: Request, api: Any, confirm: bool) -> Path | None:
    path = r.path(cfg.paths.raw)
    if path.exists():
        log.info("cache_hit", path=str(path))
        return path
    cost = quote(cfg, r, api)
    ledger = SpendLedger(cfg.databento.spend_ledger, cfg.databento.budget_usd)
    ledger.check(cost)
    if not confirm:
        log.info(
            "dry_run",
            window=r.window_id,
            instrument=r.instrument,
            schema=r.schema,
            cost_usd=round(cost, 4),
        )
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    api.timeseries.get_range(**_params(cfg, r), path=str(path))
    ledger.record(r, cost)
    log.info(
        "downloaded", path=str(path), cost_usd=round(cost, 4), spent_usd=round(ledger.spent, 4)
    )
    return path


# --- conversion ---------------------------------------------------------------------------


def _char(x: object) -> str:
    """Databento stores action/side as one-byte chars; numpy may give bytes, ints or str."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode()
    if isinstance(x, (int, np.integer)):
        return chr(int(x))
    return str(x)


def _text(x: object) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def to_ticks(fixed: npt.NDArray[np.int64], tick_size: float) -> npt.NDArray[np.int64]:
    out = np.full(fixed.shape[0], NO_PRICE, dtype=np.int64)
    ok = fixed != UNDEF_PRICE
    out[ok] = np.rint(fixed[ok] * FIXED_PRICE_SCALE / tick_size).astype(np.int64)
    return out


def mbp_to_frame(records: npt.NDArray[Any], tick_size: float) -> pl.DataFrame:
    """mbp-1 or mbp-10 records to the canonical frame (prices in ticks, levels as listed)."""
    names = records.dtype.names or ()
    depth = sum(1 for nm in names if nm.startswith("bid_px_"))
    cols: dict[str, Any] = {
        "ts_event": records["ts_event"].astype(np.int64),
        "sequence": records["sequence"].astype(np.int64),
        "instrument_id": records["instrument_id"].astype(np.int64),
        "action": np.array(
            [ACTION_MAP.get(_char(a), MODIFY) for a in records["action"]], dtype=np.int8
        ),
        "side": np.array([SIDE_MAP.get(_char(s), NONE) for s in records["side"]], dtype=np.int8),
        "price": to_ticks(records["price"].astype(np.int64), tick_size),
        "size": records["size"].astype(np.int64),
    }
    for i in range(depth):
        cols[f"bid_px_{i}"] = to_ticks(records[f"bid_px_{i:02d}"].astype(np.int64), tick_size)
        cols[f"ask_px_{i}"] = to_ticks(records[f"ask_px_{i:02d}"].astype(np.int64), tick_size)
        cols[f"bid_sz_{i}"] = records[f"bid_sz_{i:02d}"].astype(np.int64)
        cols[f"ask_sz_{i}"] = records[f"ask_sz_{i:02d}"].astype(np.int64)
    return pl.DataFrame(cols)


def ohlcv_to_frame(records: npt.NDArray[Any], tick_size: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": records["ts_event"].astype(np.int64),
            "instrument_id": records["instrument_id"].astype(np.int64),
            **{
                k: to_ticks(records[k].astype(np.int64), tick_size)
                for k in ("open", "high", "low", "close")
            },
            "volume": records["volume"].astype(np.int64),
        }
    )


def read_dbn(path: Path) -> npt.NDArray[Any]:
    import databento as db

    arr: npt.NDArray[Any] = db.DBNStore.from_file(str(path)).to_ndarray()
    return arr


def _multiplier(records: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    names = records.dtype.names or ()
    if "unit_of_measure_qty" in names:
        v = records["unit_of_measure_qty"].astype(np.int64) * FIXED_PRICE_SCALE
        out: npt.NDArray[np.float64] = np.where(v > 0, v, np.nan)
        return out
    return np.full(records.shape[0], np.nan)


def definitions_frame(records: npt.NDArray[Any]) -> pl.DataFrame:
    """Tick size, raw symbol and expiry per instrument_id, from definition records."""
    return pl.DataFrame(
        {
            "instrument_id": records["instrument_id"].astype(np.int64),
            "raw_symbol": [_text(s) for s in records["raw_symbol"]],
            "tick_size": records["min_price_increment"].astype(np.int64) * FIXED_PRICE_SCALE,
            # Contract size in units of the underlying (e.g. 1000 for ZN = $100,000 face / 100):
            # one tick is worth tick_size x multiplier USD.
            "multiplier": _multiplier(records),
            "expiration_ns": records["expiration"].astype(np.int64),
        }
    ).unique("instrument_id", keep="last")
