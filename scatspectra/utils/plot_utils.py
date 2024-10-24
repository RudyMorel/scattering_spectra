from typing import Literal
import colorsys
from matplotlib.axes import Axes
import matplotlib.colors as mc


def set_same_lim(*axes: Axes, axis: Literal['x', 'y', 'both'] = 'y') -> None:
    """ Set the same limits for a list of axes. """
    if axis == "both":
        set_same_lim(*axes, axis='x')
        set_same_lim(*axes, axis='y')
        return
    get = lambda ax: ax.get_xlim() if axis == 'x' else ax.get_ylim()
    set = lambda ax, lim: ax.set_xlim(lim) if axis == 'x' else ax.set_ylim(lim)
    lim_min = min([get(ax)[0] for ax in axes])
    lim_max = max([get(ax)[1] for ax in axes])
    for ax in axes:
        set(ax, (lim_min, lim_max))


def lighten_color(color, amount: float = 0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
