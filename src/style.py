"""Style des figures matplotlib aux couleurs de l'École Centrale Casablanca (vert #02987D, gris #72777A)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

GREEN, GREY, DARK, AMBER = "#02987D", "#72777A", "#2B3134", "#D39B2A"


def apply():
    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=[GREEN, GREY, AMBER, "#C0503F", "#5BB8A6", "#A9ADB0"]),
        "axes.edgecolor": "#C9CFD2", "axes.labelcolor": DARK, "axes.titlecolor": DARK, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E6EAEC", "grid.linewidth": 0.8,
        "xtick.color": GREY, "ytick.color": GREY, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.family": "sans-serif",
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight"})
