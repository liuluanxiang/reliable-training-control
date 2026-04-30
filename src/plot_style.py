import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "step": "#7A8793",
    "cosine": "#C9A227",
    "plateau": "#5C8F6B",
    "reliability": "#B0004F",
    "gray": "#4D4D4D",
    "lightgray": "#D9D9D9",
    "midgray": "#8A8A8A",
    "axis": "#2B2B2B",
    "accent": "#B0004F",
}

LABELS = {
    "step": "Step",
    "cosine": "Cosine",
    "plateau": "Plateau",
    "reliability": "Ours",
}

METHOD_LABELS = {
    "step": "Step",
    "cosine": "Cosine",
    "plateau": "Plateau",
    "reliability": "Ours",
}

METHOD_ORDER = ["step", "cosine", "plateau", "reliability"]


def set_nature_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": False,
        "legend.frameon": False,
    })


def panel_label(ax, label):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def save_figure(fig, out_base):
    fig.tight_layout()
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=600)
    plt.close(fig)
