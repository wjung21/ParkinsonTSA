import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd

ELECTRODES_PATH = "/Users/wjung/Library/CloudStorage/OneDrive-UCSF/Documents/PhD/ParkinsonTSA/data/ds007526/sub-002/eeg/sub-002_space-CapTrak_electrodes.tsv"

# CapTrak anatomical landmarks (from coordsystem.json)
NAS = np.array([0.0, 0.09842, 0.0])
LPA = np.array([-0.09842, 0.0, 0.0])
RPA = np.array([0.09842, 0.0, 0.0])

EOG_NAMES = {"EOG1", "EOG2", "EOG3", "EOG4"}


def _load_electrodes(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["x"] != "n/a"].copy()
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)
    eeg = df[~df["name"].isin(EOG_NAMES)]
    eog = df[df["name"].isin(EOG_NAMES)]
    return eeg, eog


def plot_2d(electrodes_path=ELECTRODES_PATH, out_path="electrode_locations_2d.png"):
    """Top-down 2D projection of EEG channel coordinates."""
    eeg, eog = _load_electrodes(electrodes_path)

    head_radius = float(np.linalg.norm(NAS[:2]))
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_aspect("equal")

    # Head outline
    ax.add_patch(plt.Circle((0, 0), head_radius, color="lightgray", fill=False, linewidth=2))

    # Nose
    nw, nh = head_radius * 0.12, head_radius * 0.1
    ax.add_patch(plt.Polygon(
        [(-nw, head_radius), (nw, head_radius), (0, head_radius + nh)],
        closed=True, color="lightgray", fill=False, linewidth=2,
    ))

    # Ears
    for sign in (-1, 1):
        ax.add_patch(mpatches.Ellipse(
            (sign * head_radius, 0),
            width=head_radius * 0.06, height=head_radius * 0.12,
            color="lightgray", fill=False, linewidth=2,
        ))

    elec_r, fs = head_radius * 0.045, 6.5
    for _, row in eeg.iterrows():
        ax.add_patch(plt.Circle((row["x"], row["y"]), elec_r, color="steelblue", zorder=3))
        ax.text(row["x"], row["y"], row["name"], ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4)

    for _, row in eog.iterrows():
        ax.add_patch(plt.Circle((row["x"], row["y"]), elec_r, color="tomato", zorder=3))
        ax.text(row["x"], row["y"], row["name"], ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4)

    for label, (lx, ly) in [("NAS", NAS[:2]), ("LPA", LPA[:2]), ("RPA", RPA[:2])]:
        ax.plot(lx, ly, "k^", markersize=8, zorder=5)
        ax.text(lx, ly + head_radius * 0.04, label, ha="center", va="bottom", fontsize=8)

    ax.legend(handles=[
        mpatches.Patch(color="steelblue", label="EEG"),
        mpatches.Patch(color="tomato", label="EOG"),
    ], loc="lower right", fontsize=9)

    pad = head_radius * 0.35
    ax.set_xlim(-head_radius - pad, head_radius + pad)
    ax.set_ylim(-head_radius - pad, head_radius + nh + pad)
    ax.set_title("EEG Channel Locations — sub-002 (CapTrak, top-down view)", fontsize=13, pad=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 2D plot to {out_path}")
    plt.show()


def plot_3d(electrodes_path=ELECTRODES_PATH, out_path="electrode_locations_3d.png"):
    """3D scatter plot of EEG channel coordinates on a semi-transparent head sphere."""
    eeg, eog = _load_electrodes(electrodes_path)

    head_radius = float(np.linalg.norm(NAS))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Semi-transparent reference sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    sx = head_radius * np.outer(np.cos(u), np.sin(v))
    sy = head_radius * np.outer(np.sin(u), np.sin(v))
    sz = head_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(sx, sy, sz, color="lightgray", alpha=0.12, linewidth=0, zorder=0)

    # Anatomical landmarks — drawn last so they sit on top of any overlapping channel markers
    landmark_offset = head_radius * 0.12
    for label, pt in [("NAS", NAS), ("LPA", LPA), ("RPA", RPA)]:
        # Offset the label radially outward from the origin so it clears nearby electrodes
        direction = pt / np.linalg.norm(pt)
        lx, ly, lz = pt + direction * landmark_offset
        ax.scatter(*pt, color="crimson", s=160, marker="*", zorder=7, edgecolors="darkred", linewidths=0.5)
        ax.text(lx, ly, lz, label, fontsize=9, color="crimson", fontweight="bold",
                ha="center", va="center", zorder=8)

    fs = 6.0
    text_gap = head_radius * 0.025  # radial gap between dot and label
    # EEG electrodes
    for _, row in eeg.iterrows():
        x, y, z = row["x"], row["y"], row["z"]
        ax.scatter(x, y, z, color="steelblue", s=120, zorder=4, depthshade=False)
        r = np.sqrt(x**2 + y**2 + z**2)
        tx, ty, tz = x + x / r * text_gap, y + y / r * text_gap, z + z / r * text_gap
        ax.text(tx, ty, tz, row["name"], fontsize=fs, color="black", zorder=5)

    # EOG electrodes
    for _, row in eog.iterrows():
        x, y, z = row["x"], row["y"], row["z"]
        ax.scatter(x, y, z, color="tomato", s=120, zorder=4, depthshade=False)
        r = np.sqrt(x**2 + y**2 + z**2)
        tx, ty, tz = x + x / r * text_gap, y + y / r * text_gap, z + z / r * text_gap
        ax.text(tx, ty, tz, row["name"], fontsize=fs, color="black", zorder=5)

    ax.set_xlabel("X (left→right, m)")
    ax.set_ylabel("Y (back→front, m)")
    ax.set_zlabel("Z (down→up, m)")
    ax.set_title("EEG Channel Locations — sub-002 (CapTrak, 3D)", fontsize=13, pad=14)

    legend_handles = [
        mpatches.Patch(color="steelblue", label="EEG electrode"),
        mpatches.Patch(color="tomato", label="EOG electrode"),
        mpatches.Patch(color="crimson", label="Anatomical landmark (NAS/LPA/RPA)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved 3D plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_2d()
    plot_3d()
