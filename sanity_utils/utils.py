from collections import namedtuple
from matplotlib import pyplot as plt
import os


Stats = namedtuple("Stats", ["min", "max", "mean", "var", "count"])


def custom_plt_style():
    """
    Configures matplotlib.pyplot to use a specific style.
    """
    plt.style.use([
        "ggplot",
        os.path.join(os.path.dirname(__file__), os.pardir, "figures.mplstyle")
    ])

