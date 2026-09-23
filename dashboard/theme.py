"""Shared plotting theme.

One palette for every figure, in a light and a dark variant. The colours are
a colour-vision-safe set; only the three slots that are safe when any series
can sit beside any other are used for categorical comparisons.

Every figure here is drawn by matplotlib from the metrics files the training
runs wrote. Nothing is hand-drawn, so a figure cannot disagree with the
numbers beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

DPI = 160


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    grid: str
    border: str
    ink: str
    ink_2: str
    ink_3: str
    series: tuple[str, str, str]
    highlight: str
    muted: str


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    grid="#e7e9ed",
    border="#d8dbe0",
    ink="#16181d",
    ink_2="#454951",
    ink_3="#6b7078",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    highlight="#1d5fae",
    muted="#b6b4ad",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    grid="#2f2f2d",
    border="#3a3a37",
    ink="#ffffff",
    ink_2="#c3c2b7",
    ink_3="#95948b",
    series=("#3987e5", "#d95926", "#199e70"),
    highlight="#3987e5",
    muted="#5c5b55",
)

THEMES = (LIGHT, DARK)


def style_axes(ax, theme: Theme, xgrid: bool = False, ygrid: bool = True) -> None:
    """House style: recessive grid, no box, ticks off, labels in ink."""
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    # matplotlib enables the grid if line properties are passed, even with
    # False, so the properties only go in when the grid is actually wanted.
    if ygrid:
        ax.grid(True, axis="y", color=theme.grid, linewidth=0.8)
    else:
        ax.grid(False, axis="y")
    if xgrid:
        ax.grid(True, axis="x", color=theme.grid, linewidth=0.8)
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.border)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.ink_3, labelsize=9, length=0)
    ax.xaxis.label.set_color(theme.ink_3)
    ax.yaxis.label.set_color(theme.ink_3)
    ax.title.set_color(theme.ink)


def legend(ax, theme: Theme, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(theme.ink_2)
    return leg


def bar_labels(ax, bars, values, theme: Theme, fmt: str = "{:.3f}",
               pad_frac: float = 0.02) -> None:
    """Direct labels on bars, so the chart is readable without gridline maths."""
    top = max(values) if values else 1.0
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * pad_frac,
            fmt.format(value),
            ha="center", va="bottom",
            color=theme.ink_2, fontsize=9,
        )


def finish(fig) -> bytes:
    """Render to PNG bytes; nothing here writes to disk."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
