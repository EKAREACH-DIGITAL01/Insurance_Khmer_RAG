"""Editorial style shared by the four measured-result figures.

The figures are built on a fixed 12 x 6 inch canvas.  In particular, ``save``
does not use ``bbox_inches="tight"``: a tight bounding box changes the output
aspect ratio according to whichever annotation extends furthest.

Typography deliberately avoids synthetic character tracking and thin spaces.
Charter is present on the target Mac and contains every non-ASCII character
used by these scripts (notably en dash and middle dot).
"""
from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's font cache somewhere writable when the scripts are run from
# a restricted environment.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/khmer-rag-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Canvas ---------------------------------------------------------------------
FIGSIZE = (12.0, 6.0)
DPI = 300
PAPER = "#FFFFFF"


# Ink and colour --------------------------------------------------------------
INK = "#172A33"
INK_SOFT = "#52636B"
INK_FAINT = "#7F8B90"
RULE = "#CBD3D5"
GRID = "#E2E7E8"
WASH = "#F4F6F6"
CACHE_WASH = "#F3EFF6"

# Muted descendants of a colourblind-safe blue / vermilion / teal / purple
# palette.  Charts that put hues on the same axes also differ in line style,
# marker shape, or hatch, so they remain separable in monochrome.
BLUE = "#2F6F91"
VERMILION = "#C75B3D"
TEAL = "#146B70"
PURPLE = "#735A86"
LIGHT_GREY = "#B7BEC0"


# Type -----------------------------------------------------------------------
SERIF = ["Charter", "Iowan Old Style", "Palatino", "DejaVu Serif"]
SANS = ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"]

SZ_TITLE = 19
SZ_SUBTITLE = 10.5
SZ_PANEL_LETTER = 13
SZ_PANEL_TITLE = 12
SZ_PANEL_DETAIL = 9
SZ_AXIS = 10.5
SZ_TICK = 9
SZ_NOTE = 8.4
SZ_LEGEND = 9
SZ_METRIC_LABEL = 7.4
SZ_METRIC_VALUE = 13


def use() -> None:
    """Apply the report-wide plotting defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF,
            "font.sans-serif": SANS,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_SOFT,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": SZ_TICK,
            "ytick.labelsize": SZ_TICK,
            "axes.linewidth": 0.8,
            "axes.titlecolor": INK,
            "axes.titlesize": SZ_PANEL_TITLE,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": (0, (2, 3)),
            "legend.frameon": False,
            "legend.fontsize": SZ_LEGEND,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "patch.linewidth": 0.6,
        }
    )


def figure(title: str, subtitle: str) -> plt.Figure:
    """Create the fixed canvas and its editorial heading."""
    fig = plt.figure(figsize=FIGSIZE, facecolor=PAPER)
    fig.text(
        0.065,
        0.947,
        title,
        ha="left",
        va="top",
        fontsize=SZ_TITLE,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.066,
        0.892,
        subtitle,
        ha="left",
        va="top",
        fontsize=SZ_SUBTITLE,
        color=INK_SOFT,
    )
    fig.add_artist(
        Line2D(
            [0.065, 0.975],
            [0.852, 0.852],
            transform=fig.transFigure,
            color=RULE,
            lw=0.8,
        )
    )
    fig.add_artist(
        Line2D(
            [0.065, 0.145],
            [0.852, 0.852],
            transform=fig.transFigure,
            color=TEAL,
            lw=2.4,
        )
    )
    return fig


def axes(ax: plt.Axes, grid: str | None = "y") -> plt.Axes:
    """Style a data axes with quiet, print-friendly furniture."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(pad=4)
    return ax


def panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    detail: str | None = None,
    *,
    y: float = 1.08,
) -> None:
    """Set a compact panel letter, title, and optional second-line detail."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SZ_PANEL_LETTER,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.072,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SZ_PANEL_TITLE,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    if detail:
        ax.text(
            0.072,
            y - 0.01,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SZ_PANEL_DETAIL,
            color=INK_FAINT,
            clip_on=False,
        )


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write an exact 3600 x 1800 pixel PNG without aspect-ratio mutation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    assert abs(width / height - 2.0) < 1e-12
    fig.savefig(path, dpi=DPI, facecolor=PAPER)
    return path
