"""论文图形的共享配色、版式、标注与导出规范。"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PALETTE = {
    "step": "#7A8793",
    "cosine": "#C9A227",
    "plateau": "#5C8F6B",
    "reliability": "#B0004F",
    "reliability_full": "#B0004F",
    "reliability_pb_only": "#D66BA0",
    "reliability_pd_only": "#8B5FBF",
    "reliability_pd_full": "#B0004F",
    "reliability_pd_v_only": "#4B9CD3",
    "reliability_pd_v_h": "#5C8F6B",
    "reliability_pd_v_u": "#C9A227",
    "reliability_pd_h_u": "#8B5FBF",
    "reliability_no_smoothing": "#E07A5F",
    "reliability_no_ema": "#E07A5F",
    "reliability_val_loss_only": "#4B9CD3",
    "reliability_monitor_only": "#2F7F7F",
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
    "reliability_full": "Ours",
    "reliability_pb_only": "PB-only",
    "reliability_pd_only": "PD-only",
    "reliability_pd_full": "Full (V+H+U)",
    "reliability_pd_v_only": "V only",
    "reliability_pd_v_h": "V+H",
    "reliability_pd_v_u": "V+U",
    "reliability_pd_h_u": "H+U",
    "reliability_no_smoothing": "No smoothing",
    "reliability_no_ema": "No smoothing",
    "reliability_val_loss_only": "Val-loss only",
    "reliability_monitor_only": "Monitor-only",
}

METHOD_LABELS = {
    "step": "Step",
    "cosine": "Cosine",
    "plateau": "Plateau",
    "reliability": "Ours",
    "reliability_full": "Ours",
    "reliability_pb_only": "PB-only",
    "reliability_pd_only": "PD-only",
    "reliability_pd_full": "Full (V+H+U)",
    "reliability_pd_v_only": "V only",
    "reliability_pd_v_h": "V+H",
    "reliability_pd_v_u": "V+U",
    "reliability_pd_h_u": "H+U",
    "reliability_no_smoothing": "No smoothing",
    "reliability_no_ema": "No smoothing",
    "reliability_val_loss_only": "Val-loss only",
    "reliability_monitor_only": "Monitor-only",
}

METHOD_ORDER = [
    "step", "cosine", "plateau", "reliability", "reliability_full",
    "reliability_pb_only", "reliability_pd_only", "reliability_no_smoothing",
    "reliability_no_ema", "reliability_val_loss_only", "reliability_monitor_only",
    "reliability_pd_full", "reliability_pd_v_only", "reliability_pd_v_h",
    "reliability_pd_v_u", "reliability_pd_h_u",
]


def set_nature_style():
    """应用面向论文双栏排版的 Matplotlib 全局参数。"""

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
        "figure.constrained_layout.use": True,
    })


def panel_label(ax, label):
    """在坐标轴左上角添加面板字母。"""

    add_panel_label(ax, label)


def add_panel_label(ax, label, x=-0.12, y=1.13):
    """以可调位置添加面板标签。"""

    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="top", ha="left", clip_on=False,
    )


def _ordered_legend(handles, labels):
    """按论文约定的方法顺序整理图例项目。"""

    order = {METHOD_LABELS.get(k, k): i for i, k in enumerate(METHOD_ORDER)}
    pairs = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        pairs.append((order.get(label, 100 + len(pairs)), handle, label))
    pairs.sort(key=lambda item: item[0])
    return [p[1] for p in pairs], [p[2] for p in pairs]


def place_legend_safely(ax, preferred="best", outside_if_needed=True, ncol=1, fontsize=6.5):
    """选择低遮挡图例位置，必要时把图例移到坐标轴外。"""

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _ordered_legend(handles, labels)
    if not handles:
        return None

    crowded = outside_if_needed == "always" or len(handles) > 3 or len(ax.lines) + len(ax.collections) > 5
    if outside_if_needed and crowded:
        return ax.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, frameon=False, fontsize=fontsize, ncol=1,
            handlelength=1.4, handletextpad=0.5, labelspacing=0.35,
        )
    return ax.legend(
        handles, labels, loc=preferred, frameon=False, fontsize=fontsize,
        ncol=ncol, handlelength=1.4, handletextpad=0.5, labelspacing=0.35,
        borderaxespad=0.6,
    )


def _axes_points(ax):
    """提取坐标轴内已有线、散点和柱形的显示坐标。"""

    points = []
    to_axes = ax.transAxes.inverted()
    for line in ax.lines:
        x = line.get_xdata(orig=False)
        y = line.get_ydata(orig=False)
        if len(x) and len(y):
            xy = ax.transData.transform(list(zip(x, y)))
            points.extend(to_axes.transform(xy))
    for collection in ax.collections:
        try:
            offsets = collection.get_offsets()
        except Exception:
            continue
        if offsets is not None and len(offsets):
            xy = ax.transData.transform(offsets)
            points.extend(to_axes.transform(xy))
    return points


def _least_crowded_annotation_loc(ax):
    """根据已有图元密度选择统计标注角落。"""

    points = [(float(x), float(y)) for x, y in _axes_points(ax) if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0]
    if not points:
        return "top-right"
    corners = {
        "top-right": lambda x, y: x > 0.50 and y > 0.55,
        "top-left": lambda x, y: x < 0.50 and y > 0.55,
        "bottom-right": lambda x, y: x > 0.50 and y < 0.45,
        "bottom-left": lambda x, y: x < 0.50 and y < 0.45,
    }
    preference = {"top-right": 0, "top-left": 1, "bottom-right": 2, "bottom-left": 3}
    scores = []
    for name, contains in corners.items():
        count = sum(1 for x, y in points if contains(x, y))
        scores.append((count, preference[name], name))
    return min(scores)[2]


def add_stat_annotation(ax, text, loc="top-right", pad=0.035, fontsize=6.5):
    """添加带背景的统计文本，并支持自动低密度位置。"""

    if not text:
        return None
    if loc == "auto":
        loc = _least_crowded_annotation_loc(ax)
    positions = {
        "top-right": (1.0 - pad, 1.0 - pad, "right", "top"),
        "top-left": (pad, 1.0 - pad, "left", "top"),
        "top-center": (0.5, 1.0 - pad, "center", "top"),
        "bottom-right": (1.0 - pad, pad, "right", "bottom"),
        "bottom-left": (pad, pad, "left", "bottom"),
    }
    x, y, ha, va = positions.get(loc, positions["top-right"])
    return ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
        color=PALETTE["axis"], clip_on=False,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.86),
        zorder=8,
    )


def draw_schematic_box(ax, xy, width, height, text, edgecolor=None, facecolor="#F8F8F8", fontsize=8):
    """绘制算法流程示意图中的统一节点框。"""

    x, y = xy
    edgecolor = edgecolor or PALETTE["midgray"]
    box = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.035",
        transform=ax.transAxes, linewidth=1.0, edgecolor=edgecolor,
        facecolor=facecolor, clip_on=False, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, color=PALETTE["axis"],
        clip_on=False, zorder=3,
    )
    return box


def draw_arrow_between_boxes(ax, start_box, end_box, y_offset=0.0):
    """在两个流程节点的边界之间绘制箭头。"""

    sx = start_box.get_x() + start_box.get_width() + 0.012
    sy = start_box.get_y() + start_box.get_height() / 2 + y_offset
    ex = end_box.get_x() - 0.012
    ey = end_box.get_y() + end_box.get_height() / 2 + y_offset
    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey), transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=9, linewidth=1.0, color=PALETTE["axis"],
        shrinkA=0, shrinkB=0, clip_on=False, zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def draw_schematic_sequence(ax, title, nodes, highlight_last=True):
    """绘制水平算法步骤序列，并可突出最终控制动作。"""

    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=12)
    n = len(nodes)
    gap = 0.055 if n >= 4 else 0.08
    left = 0.05
    total_gap = gap * (n - 1)
    width = (0.90 - total_gap) / n
    width = min(width, 0.20)
    total_width = n * width + total_gap
    left = (1.0 - total_width) / 2
    height = 0.24
    y = 0.40
    boxes = []
    for i, label in enumerate(nodes):
        x = left + i * (width + gap)
        edge = PALETTE["accent"] if (highlight_last and i == n - 1) or "PB" in label or "PD" in label or "LR" in label else PALETTE["midgray"]
        boxes.append(draw_schematic_box(ax, (x, y), width, height, label, edgecolor=edge))
    for start, end in zip(boxes[:-1], boxes[1:]):
        draw_arrow_between_boxes(ax, start, end)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return boxes


def finalize_figure_layout(fig, out_base):
    """完成布局并导出 PDF 与高分辨率 PNG。"""

    try:
        fig.set_constrained_layout_pads(w_pad=0.035, h_pad=0.045, wspace=0.10, hspace=0.12)
    except Exception:
        pass
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close(fig)


def save_figure(fig, out_base):
    """统一保存并关闭案例研究图，避免批量绘图占用内存。"""

    finalize_figure_layout(fig, out_base)
