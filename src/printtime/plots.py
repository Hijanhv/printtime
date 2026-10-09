"""Shared matplotlib styling for report figures.

Colors follow a fixed categorical order and are keyed to the entity (a model
or strategy keeps its color in every chart). Diverging heatmaps use a blue/red
pair with a neutral gray midpoint. Grids and axes are kept quiet so the data
carries the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]

ENTITY_COLORS = {
    # instruments keep one colour in every figure
    "ZT": CATEGORICAL[0],
    "ZF": CATEGORICAL[6],
    "ZN": CATEGORICAL[1],
    "ZB": CATEGORICAL[2],
    "ES": CATEGORICAL[3],
    "GC": CATEGORICAL[4],
    "CL": CATEGORICAL[5],
    "6E": CATEGORICAL[7],
    "6J": "#8a8984",
    # release vs control
    "release": CATEGORICAL[1],
    "control": "#8a8984",
}

DIVERGING = LinearSegmentedColormap.from_list(
    "halftick_div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32d2c"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("halftick_seq", ["#cde2fb", "#3987e5", "#0d366b"])


def color(entity: str, fallback_index: int = 0) -> str:
    return ENTITY_COLORS.get(entity, CATEGORICAL[fallback_index % len(CATEGORICAL)])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "sans-serif",
        }
    )


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


apply_style()
