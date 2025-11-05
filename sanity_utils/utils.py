from collections import namedtuple
from matplotlib import pyplot as plt
import os


Stats = namedtuple("Stats", ["min", "max", "mean", "var", "count"])


_style_dict = {
    "axes.labelsize" : 16,
    "xtick.labelsize" : 16,
    "ytick.labelsize" : 16,
    "legend.fontsize" : 16,
    "legend.frameon" : True,
    "legend.framealpha" : 0.8,
    "figure.titlesize" : 24,
    "figure.titleweight" : "bold",
    "legend.loc" : "upper right",
    "savefig.dpi" : "figure",
    "figure.dpi" : 800,
    "savefig.bbox" : "tight",
    "savefig.transparent" : False
}


def custom_plt_style():
    """
    Configures matplotlib.pyplot to use a specific style.
    """
    plt.style.use([
        "ggplot",
        _style_dict
    ])

