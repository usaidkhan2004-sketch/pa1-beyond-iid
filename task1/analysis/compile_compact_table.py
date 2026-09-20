import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

RESULTS_DIR = REPO_ROOT / "task1" / "results"


def compile_compact_table():
    with open(RESULTS_DIR / "clean_baseline_metrics.json", "r", encoding="utf-8") as f:
        clean = json.load(f)
    with open(RESULTS_DIR / "color_bias_metrics.json", "r", encoding="utf-8") as f:
        color = json.load(f)
    with open(RESULTS_DIR / "patch_shuffle_metrics.json", "r", encoding="utf-8") as f:
        shuffle = json.load(f)

    models = ["resnet50_probe", "vit_b16_probe", "clip_vit_b32_probe", "clip_vit_b32_zero_shot"]
    model_labels = {
        "resnet50_probe": "ResNet-50 (Linear Probe)",
        "vit_b16_probe": "ViT-B/16 (Linear Probe)",
        "clip_vit_b32_probe": "CLIP ViT-B/32 (Linear Probe)",
        "clip_vit_b32_zero_shot": "CLIP ViT-B/32 (Zero-Shot)",
    }

    summary = {}

    print("\n" + "=" * 115)
    print("REQUIRED EVIDENCE 1: COMPACT INTERVENTION PERFORMANCE COMPARISON")
    print("=" * 115)
    header = f"| {'Model Architecture':<28} | {'Clean Acc':<10} | {'Grayscale Acc (Drop)':<22} | {'Hue-Rot Acc (Drop)':<22} | {'Shuffle Acc (Drop)':<22} |"
    print(header)
    print("|" + "-" * 30 + "|" + "-" * 12 + "|" + "-" * 24 + "|" + "-" * 24 + "|" + "-" * 24 + "|")

    for m in models:
        c_acc = clean[m]["top1_accuracy"]

        g_acc = color["grayscale"][m]["metrics"]["top1_accuracy"]
        g_drop = color["grayscale"][m]["relative_drop"]["acc_drop"]

        h_acc = color["hue_rotation"][m]["metrics"]["top1_accuracy"]
        h_drop = color["hue_rotation"][m]["relative_drop"]["acc_drop"]

        s_acc = shuffle[m]["metrics"]["top1_accuracy"]
        s_drop = shuffle[m]["relative_drop"]["acc_drop"]
        s_cons = shuffle[m]["consistency"]

        summary[m] = {
            "clean_acc": c_acc,
            "grayscale_acc": g_acc,
            "grayscale_drop": g_drop,
            "hue_acc": h_acc,
            "hue_drop": h_drop,
            "shuffle_acc": s_acc,
            "shuffle_drop": s_drop,
            "shuffle_consistency": s_cons,
        }

        row = (
            f"| {model_labels[m]:<28} "
            f"| {c_acc:>7.2f}%  "
            f"| {g_acc:>6.2f}% (-{g_drop:>4.2f}%)   "
            f"| {h_acc:>6.2f}% (-{h_drop:>4.2f}%)   "
            f"| {s_acc:>6.2f}% (-{s_drop:>4.2f}%)   |"
        )
        print(row)

    print("=" * 115 + "\n")

    out_json = RESULTS_DIR / "compact_intervention_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary table saved to: {out_json}\n")


if __name__ == "__main__":
    compile_compact_table()