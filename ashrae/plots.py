"""The look shared by the figures of every section."""
INK, TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def style_axis(ax):
    """The shared look: light surface, horizontal grid only, no frame but the bottom spine, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0, pad=6)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
