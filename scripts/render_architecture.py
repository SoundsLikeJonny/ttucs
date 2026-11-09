"""Render the README architecture diagram (docs/assets/architecture.png).

Style follows the conventions of published AI architecture diagrams: left-to-right data flow,
grouped stages, labelled arrows carrying the payload, and a routed side-path for the slate
fields that bypass the classifier. Run from the repo root: `python scripts/render_architecture.py`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# palette: quiet paper tones, one accent per stage family
BG = "#fdfdfc"
STAGE = ["#eef2f7", "#eaf3ec", "#f7f0e4", "#f3ecf5"]
EDGE = ["#5b7ea6", "#5f9468", "#b3924e", "#8a6aa0"]
TEXT = "#2b3440"
ARROW = "#8a929c"

BOX = dict(boxstyle="round,pad=0.02", linewidth=1.4)


def box(ax, x, y, w, h, title, lines, family, title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=STAGE[family], edgecolor=EDGE[family], **BOX))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=TEXT)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.088 - i * 0.043, line, ha="center", va="top",
                fontsize=7.9, color=TEXT)


def arrow(ax, x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=ARROW, shrinkA=1, shrinkB=1))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, label, ha="center", va="bottom",
                fontsize=7.3, color="#5c6570", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.3), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, height = 0.34, 0.50  # main flow band: y 0.34 .. 0.84

    # ---- stage 1: speech to text
    box(ax, 0.015, top, 0.135, height, "Spoken slate",
        ['"exterior, parking lot,', 'ambience nearby traffic,', "i'm using an sennheiser",
         'm k eighty forty..."', "", "speech-to-text output", "(words, no audio)"], 0)

    # ---- stage 2: parser
    box(ax, 0.185, top, 0.205, height, "Slate parser   ttucs.slate",
        ["rule-based + gear lexicon,", "number-word normalisation",
         "mic / recorder / ext -> slate fields", "remaining clauses -> description"], 1)

    # ---- stage 3: model
    box(ax, 0.435, top, 0.225, height, "Classifier   ttucs.models  bow",
        ["hashed word uni/bi-grams", "+ char 3-5-grams   (131,072 x 256)",
         "EmbeddingBag -> residual MLP", "CatID head  +  Category head"],
        2)

    # ---- stage 4: outputs
    box(ax, 0.705, top, 0.28, height, "Prediction",
        ["CatID  AMBTraf     confidence  0.42",
         "decision  confident    top-5 alternatives",
         "filename  AMBTraf-EXT_Ambience Nearby...",
         "keywords  +  full slate fields"], 3)

    # ---- main-flow arrows with payloads
    mid = top + height / 2
    arrow(ax, 0.152, mid, 0.183, mid, "text")
    arrow(ax, 0.392, mid, 0.433, mid, "description")
    arrow(ax, 0.662, mid, 0.703, mid, "logits")

    # ---- calibration strip below the classifier
    box(ax, 0.435, 0.028, 0.225, 0.19, "Calibration",
        ["temperature scaling on validation",
         "decision: confident / category_fallback / unknown"], 2, title_size=9.5)
    arrow(ax, 0.5475, top, 0.5475, 0.222, "temperature")

    # ---- the hard rule: slate fields bypass the classifier into the prediction
    byp = 0.245
    ax.plot([0.2875, 0.2875], [top, byp], color="#5f9468", linewidth=1.3)
    ax.plot([0.2875, 0.845], [byp, byp], color="#5f9468", linewidth=1.3)
    ax.add_patch(FancyArrowPatch((0.845, byp), (0.845, top - 0.004), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.3, color="#5f9468",
                                 shrinkA=0, shrinkB=0))
    ax.text(0.695, byp + 0.012, "slate fields - returned, never classified",
            ha="center", va="bottom", fontsize=7.8, color="#3f6b46", style="italic")

    fig.savefig("docs/assets/architecture.png", bbox_inches="tight", facecolor=BG)
    print("wrote docs/assets/architecture.png")


if __name__ == "__main__":
    main()
