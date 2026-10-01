"""Generate PNG README assets that GitHub renders reliably (no SVG gradients)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "readme-assets"
OUT.mkdir(parents=True, exist_ok=True)


def hero_banner() -> None:
    fig, ax = plt.subplots(figsize=(12, 2.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.text(
        0.4,
        1.85,
        "Whisper Tiny - Dual-Scale Attention",
        color="#e2e8f0",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.4,
        1.35,
        "Audio-native end-of-turn detection for Hindi and English - under 50 ms CPU inference",
        color="#94a3b8",
        fontsize=10,
        va="top",
    )
    heights = [0.55, 0.7, 0.5, 0.75, 0.6, 0.68, 0.45, 0.72]
    for index, height in enumerate(heights):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.5 + index * 0.22, 0.35),
                0.12,
                height,
                boxstyle="round,pad=0.02",
                facecolor="#6366f1",
                edgecolor="none",
                alpha=0.85,
            )
        )
    ax.plot([2.6, 11.2], [0.55, 0.55], color="#334155", lw=1, linestyle="--")
    ax.add_patch(plt.Circle((11.2, 0.55), 0.06, color="#22d3ee"))
    ax.text(11.35, 0.55, "COMPLETE", color="#22d3ee", fontsize=9, fontweight="bold", va="center")
    plt.tight_layout(pad=0)
    fig.savefig(OUT / "hero_banner.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close()


def logo() -> None:
    fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.05),
            0.9,
            0.9,
            boxstyle="round,pad=0.04",
            facecolor="#0f172a",
            edgecolor="#6366f1",
            linewidth=2,
        )
    )
    theta = np.linspace(0.3 * np.pi, 1.7 * np.pi, 80)
    ax.plot(0.5 + 0.32 * np.cos(theta), 0.5 + 0.32 * np.sin(theta), color="#818cf8", lw=3)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.62, 0.35),
            0.12,
            0.35,
            boxstyle="round,pad=0.01",
            facecolor="#1e293b",
            edgecolor="#38bdf8",
            linewidth=1,
        )
    )
    wave_t = np.linspace(0, 4 * np.pi, 200)
    ax.plot(0.25 + 0.15 * wave_t / (4 * np.pi), 0.35 + 0.08 * np.sin(wave_t * 2), color="#2dd4bf", lw=2)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.04, color="#22d3ee"))
    plt.tight_layout(pad=0)
    fig.savefig(OUT / "logo.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close()


def metrics_strip() -> None:
    fig, ax = plt.subplots(figsize=(11, 1.1), dpi=150)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 11, 1, boxstyle="round,pad=0.02", facecolor="#1e293b", edgecolor="none"))
    items = [
        ("F1", "0.7399", "#22d3ee"),
        ("FALSE CUTOFF", "4.72%", "#e2e8f0"),
        ("AUROC", "0.9361", "#e2e8f0"),
        ("E2E p50 LATENCY", "47.36 ms", "#e2e8f0"),
        ("INT8 SIZE", "10.16 MiB", "#a78bfa"),
    ]
    xs = [0.3, 2.2, 4.2, 6.0, 8.4]
    for (label, value, color), x in zip(items, xs, strict=True):
        ax.text(x, 0.72, label, color="#64748b", fontsize=8, va="top")
        ax.text(x, 0.38, value, color=color, fontsize=14, fontweight="bold", va="top")
    plt.tight_layout(pad=0)
    fig.savefig(OUT / "metrics_strip.png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close()


def main() -> None:
    hero_banner()
    logo()
    metrics_strip()
    print("Wrote", OUT / "hero_banner.png", OUT / "logo.png", OUT / "metrics_strip.png")


if __name__ == "__main__":
    main()
