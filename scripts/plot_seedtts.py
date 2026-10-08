"""Render the README benchmark figure (requires matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]

# Spread the small-model cluster while keeping larger models separated.
PARAM_SCALE_POWER = 0.6

# Labels sit next to their marker with no connector lines, so every offset has to
# leave the label unambiguously attached to one point.
LABEL_OFFSETS = {
    # name: (dx points, dy points, horizontal alignment)
    "PL-Flow": (11, 2, "left"),
    "ZipVoice": (9, 0, "left"),
    "F5-TTS": (9, 0, "left"),
    "MegaTTS 3": (9, 0, "left"),
    "DiTAR": (9, 3, "left"),
    "CosyVoice 2": (9, -2, "left"),
    "CosyVoice 3 (1.5B)": (9, 0, "left"),
    "VoxCPM 2": (9, 0, "left"),
    "FireRedTTS3-Base": (-9, 0, "right"),
    "dots.tts (SOAR)": (9, 0, "left"),
    "LongCat-AudioDiT": (-9, 0, "right"),
}

X_TICKS = [0.1, 0.2, 0.5, 1, 2, 3.5]
X_TICK_LABELS = ["0.1", "0.2", "0.5", "1", "2", "3.5"]


def main():
    report = json.loads((ROOT / "assets/seedtts_eval.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.edgecolor": "#a6afb9",
            "axes.labelcolor": "#263242",
            "text.color": "#263242",
            "xtick.color": "#526171",
            "ytick.color": "#526171",
            "svg.fonttype": "none",
            "svg.hashsalt": "pl-flow-seedtts",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.20, top=0.81)
    fig.text(0.10, 0.94, "Speaker similarity vs. model size", fontsize=18, weight="semibold")
    fig.text(0.10, 0.886, "Seed-TTS-Eval · Chinese main set (test-zh) · WavLM-SV", color="#667483")

    power = PARAM_SCALE_POWER
    ax.set_xscale("function", functions=(lambda x: x**power, lambda x: x ** (1 / power)))
    ax.set_xlim(0.10, 3.75)
    ax.set_ylim(73.9, 85.4)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.set_xticklabels(X_TICK_LABELS)
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(color="#e3e8ed", linewidth=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.tick_params(length=3, width=0.7)
    ax.set_xlabel(
        f"Primary generation parameters (billions; power scale, p = {power})", labelpad=10
    )
    ax.set_ylabel("Speaker similarity (SIM × 100) ↑", labelpad=10)

    for model in report["models"]:
        name = model["model"]
        x = model["parameters"] / 1e9
        y = model["metrics"]["test-zh"]["sim_percent"]
        ours = name == "PL-Flow"
        color = "#175a9c" if ours else "#738698"
        ax.scatter(
            x, y, s=150 if ours else 48, color=color, edgecolor="white", linewidth=0.8, zorder=3
        )
        dx, dy, ha = LABEL_OFFSETS[name]
        label = f"PL-Flow\n{x:.3f}B · {y:.2f} SIM" if ours else name
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11 if ours else 9.3,
            weight="bold" if ours else "normal",
            color=color if ours else "#3e4d5c",
            ha=ha,
            va="center",
            zorder=4,
        )
    fig.text(
        0.10,
        0.065,
        "Published results; separate codecs and reference encoders excluded from parameter counts.",
        fontsize=8.5,
        color="#6c7885",
    )
    fig.text(
        0.10,
        0.032,
        "Model-specific parameter scopes and score sources: assets/seedtts_eval.json",
        fontsize=8.5,
        color="#6c7885",
    )
    for extension in ("svg", "png"):
        path = ROOT / f"assets/seedtts_zh_sim.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(path, dpi=240, metadata=metadata)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
