"""Extract results from all training notebooks."""

import json
import re
from pathlib import Path

notebooks = [
    "model_training_v2.ipynb",
    "model_training_v2_grouped.ipynb",
    "model_training_v3_grouped.ipynb",
    "model_training_v3_1_grouped.ipynb",
    "model_training_v4_sample_level.ipynb",
    "model_training_v5.ipynb",
    "model_training_v6.ipynb",
    "model_training_v6_only_two_grps.ipynb",
    "model_training_v7.ipynb",
]

for nb_name in notebooks:
    nb_path = Path("notebooks") / nb_name
    if not nb_path.exists():
        print(f"=== SKIPPED: {nb_name} (not found) ===")
        continue

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"NOTEBOOK: {nb_name}")
    print(sep)

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if any(
                k in src.lower()
                for k in ["result", "conclusion", "next step", "finding"]
            ):
                print(f"  Cell {i + 1} (markdown):")
                for line in src.split("\n")[:15]:
                    print(f"    {line}")
                print()

        if cell["cell_type"] == "code" and "outputs" in cell:
            for o in cell["outputs"]:
                data = o.get("data", {})
                if "text/html" in data:
                    html = "".join(data["text/html"])
                    if any(k in html.lower() for k in ["accuracy", "f1", "balanced"]):
                        text = re.sub(r"<[^>]+>", " ", html)
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) > 50:
                            print(f"  Cell {i + 1} (table):")
                            print(f"    {text[:2000]}")
                            print()

                if "text" in o:
                    text = "".join(o["text"])
                    keywords = [
                        "best",
                        "f1",
                        "balanced_acc",
                        "macro",
                        "precision",
                        "recall",
                        "accuracy",
                    ]
                    if any(k in text.lower() for k in keywords):
                        lines = text.split("\n")
                        relevant = [
                            l for l in lines if any(k in l.lower() for k in keywords)
                        ]
                        if relevant:
                            print(f"  Cell {i + 1} (text output):")
                            for l in relevant[:30]:
                                print(f"    {l}")
                            print()
