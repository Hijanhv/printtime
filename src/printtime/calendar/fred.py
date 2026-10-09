"""FRED release calendar: find release IDs by name, then fetch release dates.

The spec forbids hard-coding release IDs or dates. `find_release_id` searches
FRED's list of releases by name and insists on exactly one match, and
`release_dates` asks FRED for every date that release was published inside
the sample. The client takes an injectable httpx transport so tests can run
against a fake FRED with no network and no key.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import httpx

from printtime.log import get_logger

log = get_logger(__name__)


class FredError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    id: int
    name: str


class FredClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._key = api_key
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/", timeout=timeout_s, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FredClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        query = {"api_key": self._key, "file_type": "json", **params}
        r = self._client.get(path, params=query)
        if r.status_code != 200:
            # Never include the request URL: it carries the API key.
            raise FredError(f"FRED {path} returned HTTP {r.status_code}: {r.text[:200]}")
        data: dict[str, Any] = r.json()
        return data

    def releases(self) -> list[Release]:
        out: list[Release] = []
        offset = 0
        while True:
            page = self._get("releases", limit=1000, offset=offset)
            items = page.get("releases", [])
            out += [Release(int(x["id"]), str(x["name"])) for x in items]
            offset += len(items)
            if not items or offset >= int(page.get("count", offset)):
                return out

    def find_release_id(self, name: str) -> int:
        """Exact name match first; otherwise a unique case-insensitive substring match."""
        all_releases = self.releases()
        exact = [r for r in all_releases if r.name.strip().lower() == name.strip().lower()]
        if len(exact) == 1:
            return exact[0].id
        partial = [r for r in all_releases if name.lower() in r.name.lower()]
        if len(partial) == 1:
            return partial[0].id
        found = exact or partial
        raise FredError(
            f"release name {name!r} matched {len(found)} FRED releases: "
            + ", ".join(f"{r.id} {r.name!r}" for r in found[:10])
        )

    def release_dates(self, release_id: int, start: dt.date, end: dt.date) -> list[dt.date]:
        """Every date the release was published between start and end, inclusive."""
        page = self._get(
            "release/dates",
            release_id=release_id,
            realtime_start=start.isoformat(),
            realtime_end=end.isoformat(),
            include_release_dates_with_no_data="false",
            sort_order="asc",
            limit=10000,
        )
        dates = sorted({dt.date.fromisoformat(x["date"]) for x in page.get("release_dates", [])})
        return [d for d in dates if start <= d <= end]
