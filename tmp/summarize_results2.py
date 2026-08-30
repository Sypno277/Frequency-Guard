import csv
from collections import defaultdict

def parse_ai(v):
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "ai"):
        return True
    if s in ("0", "false", "no", "real"):
        return False
    return bool(int(float(v)))

rows = list(csv.DictReader(open("data/benchmark/our_results.csv", encoding="utf-8")))
n = len(rows)
tp = tn = fp = fn = 0
by_state = defaultdict(lambda: [0,0,0,0,0])  # state: [n, tp, tn, fp, fn]
for r in rows:
    pred_ai = parse_ai(r.get("is_ai_pred"))
    truth_ai = r["label"] == "ai"
    st = r["state"]
    by_state[st][0] += 1
    if pred_ai is None:
        continue
    if truth_ai:
        if pred_ai:
            tp += 1; by_state[st][1]+=1
        else:
            fn += 1; by_state[st][4]+=1
    else:
        if pred_ai:
            fp += 1; by_state[st][3]+=1
        else:
            tn += 1; by_state[st][2]+=1

acc = (tp+tn)/n
print(f"TOTAL n={n}")
print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"  ACCURACY={acc:.3f}")
print(f"  Sensitivity (AI recall) = {tp/(tp+fn) if tp+fn else 0:.3f}")
print(f"  Specificity (real)      = {tn/(tn+fp) if tn+fp else 0:.3f}")
print(f"  False-positive rate     = {fp/(fp+tn) if fp+tn else 0:.3f}")
print(f"  False-negative rate     = {fn/(fn+tp) if fn+tp else 0:.3f}")

print("\n--- By state [n, tp, tn, fp, fn] ---")
for st in ["pristine","jpeg60","resize","screenshot"]:
    s = by_state[st]
    print(f"  {st:12s} n={s[0]:2d} tp={s[1]:2d} tn={s[2]:2d} fp={s[3]:2d} fn={s[4]:2d} acc={(s[1]+s[2])/s[0] if s[0] else 0:.3f}")

# How did our detector behave on the REAL image classes by state?
print("\n--- AI image rows (all 16) ---")
for r in rows:
    if r["label"] == "ai":
        print(f"  {r['file']:42s} {r['state']:10s} pred_ai={parse_ai(r.get('is_ai_pred'))} conf={r.get('confidence'):>6s} fake_p={r.get('fake_prob'):>7s} agree={r.get('agreement'):>6s}")
