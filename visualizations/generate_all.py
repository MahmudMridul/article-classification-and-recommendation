"""Run all visualization scripts and regenerate every figure in report/figures/.

Usage (from project root):
    python visualizations/generate_all.py

Each script saves its output to report/figures/.
Scripts that need the trained models (fig6, fig7, fig8) require the saved
classifier in models/classifier/ and the recommender index in models/recommender/.
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ  = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "fig1_system_architecture.py",
    "fig2_category_distribution.py",
    "fig3_title_length.py",
    "fig4_data_split.py",
    "fig4_pipeline_diagram.py",
    "fig5_training_curves.py",
    "fig6_confusion_matrix.py",
    "fig7_per_class_metrics.py",
    "fig8_recommendation_scores.py",
]


def main():
    ok, failed = [], []
    for script in SCRIPTS:
        path = os.path.join(VIZ, script)
        print(f"\n{'='*60}")
        print(f"Running {script} ...")
        print('='*60)
        result = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
        )
        if result.returncode == 0:
            ok.append(script)
        else:
            failed.append(script)

    print(f"\n{'='*60}")
    print(f"Done: {len(ok)}/{len(SCRIPTS)} scripts succeeded.")
    if failed:
        print("Failed scripts:")
        for s in failed:
            print(f"  - {s}")
    print('='*60)


if __name__ == "__main__":
    main()
