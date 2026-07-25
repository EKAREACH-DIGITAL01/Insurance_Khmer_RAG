"""Shared visual style for the report's result figures.

Derived from the style reference generated in output/imagegen/ -- restrained
academic-journal styling: transitional serif, letter-spaced small caps, hairline
axes, faint dashed grid, muted palette.

Palette sampled from those reference images, then corrected: the reference also
used a sage green (#93a884) next to the ochre, and that pair sits at delta-E 9.2
in normal vision and 6.1 under protanopia -- indistinguishable in print and
invisible to a colourblind reader. Dropping sage leaves four hues that clear both
gates (worst adjacent 12.4 protan / 25.5 normal). Four is all any figure here
needs.

The two cool hues read slightly grey by a strict chroma floor. That is inherent
to the muted aesthetic and is kept deliberately; the ochre's low contrast against
white is covered by the value labels drawn on those segments.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette ----------------------------------------------------------------
TEAL   = "#1a6670"
CORAL  = "#e06855"
SLATE  = "#486b8d"
OCHRE  = "#ceaa5f"
SERIES = [TEAL, CORAL, SLATE, OCHRE]

INK       = "#1c1c1a"   # titles, panel letters
INK_SOFT  = "#4a4a46"   # axis labels, tick labels, annotations
RULE      = "#c9c7c0"   # axis spines
GRID      = "#e2e0d9"   # faint dashed grid
SURFACE   = "#ffffff"
REFLINE   = "#3a3a37"   # dashed reference / median lines

# --- type -------------------------------------------------------------------
# Baskerville is present on macOS and is the closest match to the reference's
# transitional serif; the chain degrades gracefully elsewhere.
SERIF = ["Baskerville", "Hoefler Text", "Palatino", "Charter",
         "Times New Roman", "DejaVu Serif"]

SZ_SUPTITLE = 13
SZ_PANEL_L  = 15    # the big A / B / C / D
SZ_PANEL_T  = 9
SZ_AXIS_L   = 8.5
SZ_TICK     = 8.5
SZ_ANNOT    = 8
SZ_LEGEND   = 9


def use():
    """Applies the rcParams shared by every figure."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        SERIF,
        "figure.facecolor":  SURFACE,
        "axes.facecolor":    SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color":        INK,
        "axes.edgecolor":    RULE,
        "axes.labelcolor":   INK_SOFT,
        "xtick.color":       INK_SOFT,
        "ytick.color":       INK_SOFT,
        "xtick.labelsize":   SZ_TICK,
        "ytick.labelsize":   SZ_TICK,
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "grid.linestyle":    (0, (1.5, 2.5)),
        "legend.frameon":    False,
        "legend.fontsize":   SZ_LEGEND,
    })


def track(s, wide=False):
    """Fakes letter-spacing, which matplotlib has no property for, by inserting
    spaces between characters -- the small-caps voice in the style reference.

    Plain U+0020 only. A thin space (U+2009) spaces more finely but Baskerville
    has no glyph for it, and matplotlib silently substitutes a dummy box rather
    than failing, so the tracking renders as garbage."""
    return ("  " if wide else " ").join(s)


def smallcaps(s):
    """Uppercase + tracked: the reference's panel-title and axis-label voice."""
    return track(s.upper())


def panel(ax, letter=None, title=None, grid="y"):
    """Applies the shared panel furniture: hairline spines on two sides only,
    faint dashed grid behind the marks, and the reference's serif-capital panel
    letter set outside the axes to the upper left."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)
    if letter is not None:
        ax.annotate(letter, xy=(0.0, 1.13), xycoords="axes fraction",
                    ha="left", va="center", fontsize=SZ_PANEL_L, color=INK)
    if title is not None:
        # Offset clear of the panel letter, which is set at x = 0.
        ax.set_title(smallcaps(title), fontsize=SZ_PANEL_T, color=INK_SOFT,
                     pad=8, loc="left", x=0.085)
    return ax


def suptitle(fig, text, y=1.02):
    fig.suptitle(track(text.upper(), wide=True), fontsize=SZ_SUPTITLE,
                 color=INK, y=y)


def footnote(fig, text, y=-0.09):
    fig.text(0.5, y, text, ha="center", fontsize=SZ_ANNOT, color=INK_SOFT,
             linespacing=1.6)
