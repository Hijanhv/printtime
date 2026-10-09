"""Canonical order-book frame shared by every module.

One row per book update, carrying the book state *after* that update, the
same shape as Databento mbp-1 / mbp-10 records once converted. Prices are
integer ticks (tick size travels alongside as metadata), so comparisons are
exact and spreads are whole numbers.

Columns:
    ts_event          int64   exchange timestamp, ns since epoch (never ts_recv)
    action            int8    ADD / CANCEL / MODIFY / TRADE / FILL / CLEAR
    side              int8    BID / ASK / NONE; for trades, the aggressor side
    price             int64   ticks; trade price for trades
    size              int64   lots
    bid_px_{i}, ask_px_{i}    int64 ticks, level i (0 = touch)
    bid_sz_{i}, ask_sz_{i}    int64 lots
"""

from __future__ import annotations

from typing import Final

ADD: Final = 0
CANCEL: Final = 1
MODIFY: Final = 2
TRADE: Final = 3
FILL: Final = 4
CLEAR: Final = 5

BID: Final = 1
ASK: Final = -1
NONE: Final = 0

NO_PRICE: Final = -(2**40)


def level_columns(levels: int) -> list[str]:
    cols: list[str] = []
    for i in range(levels):
        cols += [f"bid_px_{i}", f"ask_px_{i}", f"bid_sz_{i}", f"ask_sz_{i}"]
    return cols


BASE_COLUMNS: Final = ("ts_event", "action", "side", "price", "size")
