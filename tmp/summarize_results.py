import csv
from collections import Counter, defaultdict
from pathlib import Path

rows = list(csv.DictReader(open("data/benchmark/our_results.csv", encoding="utf-8")))
print(f"Total rows: {len(rows)}")

# Ground truth analysis
def acc_for(sel):
    n = correct = 0
    for r in sel:
        label = r["label"]
        pred = r.get("is_ai_pred")
        if pred is None or pred == "":
            continue
        pred_ai = str(pred).lower() == "true"
        truth_ai = label == "ai"
        n += 1
        if pred_ai == truth_ai:
            correct += 1
    return correct, n

overall_c, overall_n = acc_for(rows)
print(f"\nOVERALL ACCURACY: {overall_c}/{overall_n} = {overall_c/overall_n:.3f}" if overall_n else "no data")

# Break down by state
by_state = defaultdict(list)
for r in rows:
    by_state[r["state"]].append(r)

print("\n--- Accuracy by state ---")
for state in ["pristine", "jpeg60", "resize", "screenshot"]:
    c, n = acc_for(by_state[state])
    print(f"{state:12s}: {c}/{n} = {c/n:.3f}" if n else f"{state}: none")

# Break down by label
by_label = defaultdict(list)
for r in rows:
    by_label[r["label"]].append(r)

print("\n--- Accuracy by ground-truth label ---")
for label in ["real", "ai"]:
    c, n = acc_for(by_label[label])
    print(f"{label:6s}: {c}/{n} = {c/n:.3f}" if n else f"{label}: none")

# Confusion: real falsely flagged as AI (FP) and AI flagged real (FN)
fps = [r for r in rows if r["label"] == "real" and str(r.get("is_ai_pred","")).lower() == "true"]
fns = [r for r in rows if r["label"] == "ai" and str(r.get("is_ai_pred","")).lower() == "false"]
print(f"\nFalse positives (real flagged AI): {len(fps)}")
for r in fps:
    print(f"  {r['file']} state={r['state']} conf={r.get('confidence')} fake_p={r.get('fake_prob')} agree={r.get('agreement')}")
print(f"False negatives (AI flagged real): {len(fns)}")
for r in fns:
    print(f"  {r['file']} state={r['state']} conf={r.get('confidence')} fake_p={r.get('fake_prob')} agree={r.get('agreement')}")

# Low agreement signal
low_agree = [r for r in rows if r.get("agreement") is not None and float(r["agreement"]) < 0.5]
print(f"\nLow-agreement (<0.5) rows: {len(low_agree)}")

# Screenshot-specific
shot = [r for r in rows if r["state"] == "screenshot"]
shot_conf = {r["label"]: sum(1 for r2 in shot if r2["label"]==r["label"]) for r in shot}
print(f"\nScreenshot rows: {len(shot)}")
shot_pred_ai = [r for r in shot if str(r.get("is_ai_pred","")).lower()=="true"]
print(f"  flagged-AI among screenshots: {len(shot_pred_ai)}/{len(shot)}")

# Summary for report
real_ai_0 = [r for r in rows if r["image_id"]=="ai_0_portrait"]
print(f"\nai_0_portrait (genuine generated) states:")
for r in real_ai_0:
    print(f"  {r['state']:12s} pred_ai={r.get('is_ai_pred')} conf={r.get('confidence')} fake_p={r.get('fake_prob')} agree={r.get('agreement')}")
