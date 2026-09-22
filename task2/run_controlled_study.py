"""
task2/run_controlled_study.py
Executes Step 6: Controlled Design Study on DAN alignment strength.
Tests lambda_MMD in {0.1, 1.0, 10.0}, logs metrics, and outputs the compact
comparison table required for the final report.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from task2.train_dan import train_dan


def main():
    # lambda = 1.0 is already trained and saved in dan_metrics.json
    study_lambdas = [0.1, 10.0]

    for l_val in study_lambdas:
        print("\n" + "=" * 60)
        print(f"   RUNNING DAN CONTROLLED STUDY: lambda_MMD = {l_val}")
        print("=" * 60)
        train_dan(
            lambda_mmd=l_val,
            history_filename=f"dan_lambda_{l_val}_history.json",
            metrics_filename=f"dan_lambda_{l_val}_metrics.json",
            ckpt_filename=f"dan_lambda_{l_val}_best.pth",
        )

    # Compile the compact comparative table across all three settings
    print("\n" + "=" * 80)
    print("      STEP 6: CONTROLLED DESIGN STUDY RESULTS (ALIGNMENT STRENGTH)")
    print("=" * 80)
    header = f"{'lambda_MMD':<12} | {'Mean Src Acc':<13} | {'Mean Src F1':<12} | {'Sketch Acc':<11} | {'Sketch F1':<10} | {'Domain Sep':<11}"
    print(header)
    print("-" * 80)

    summary_records = {}

    for l_val in [0.1, 1.0, 10.0]:
        fname = "dan_metrics.json" if l_val == 1.0 else f"dan_lambda_{l_val}_metrics.json"
        path = Path("task2/checkpoints") / fname
        if path.exists():
            with open(path, "r") as f:
                d = json.load(f)
            summary_records[str(l_val)] = d
            print(
                f"{l_val:<12} | "
                f"{d['mean_source_acc']:>11.2f}% | "
                f"{d['mean_source_f1']:>10.2f}% | "
                f"{d['target_acc']:>9.2f}% | "
                f"{d['target_f1']:>8.2f}% | "
                f"{d['domain_separability']:>9.2f}%"
            )

    print("=" * 80)

    # Save summary for plotting
    summary_file = Path("task2/checkpoints/controlled_study_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Summary table data saved to: {summary_file}")


if __name__ == "__main__":
    main()