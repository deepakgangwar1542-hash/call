"""STEPS 7-12: Level 0, Level 1, ML models, evaluation, Precision@K, save best model."""
import json, datetime as dt
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *

df = pd.read_csv(CLEAN_CSV)
X, y = df[FEATURES], df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
K = min(K_DEFAULT, len(yte) // 4)
print(f"Train={len(Xtr)} Test={len(Xte)} | base rate={y.mean():.3f} | K={K}")
results = {}

# ---- Level 0: always "Slight" -> score 0 for everything ----
s0 = np.zeros(len(yte))
results["Level0_always_Slight"] = dict(
    precision_at_k=0.0, note="never flags High risk -> precision@K undefined/0", roc_auc=0.5,
    pr_auc=float(yte.mean()))
print("Level 0 : never flags anything (precision@K = 0 / undefined)")

# ---- Level 1: group by speed_limit ----
grp_rate = ytr.groupby(Xtr["speed_limit"]).mean()          # serious-rate per speed limit
grp_major = (grp_rate >= 0.5).astype(int)                   # majority label per group (SRS definition)
s1 = Xte["speed_limit"].map(grp_rate).fillna(ytr.mean()).values
flagged = Xte["speed_limit"].map(grp_major).fillna(0).sum()
results["Level1_speed_limit_group"] = dict(
    precision_at_k=precision_at_k(yte, s1, K), roc_auc=float(roc_auc_score(yte, s1)),
    pr_auc=float(average_precision_score(yte, s1)),
    note=f"majority-label flags {int(flagged)} calls; ranking by group serious-rate with random tie-break")
print("Level 1 :", {k: round(v, 4) for k, v in results["Level1_speed_limit_group"].items() if k != "note"})

# ---- ML models (Step 9) ----
def pipe(clf):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES)])  # speed_limit bhi categorical
    return Pipeline([("pre", pre), ("clf", clf)])

Xtr_c, Xte_c = Xtr.astype({"speed_limit": str}), Xte.astype({"speed_limit": str})
models = {
    "LogisticRegression": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RandomForest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED)),
    "HistGradientBoosting": pipe(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=SEED)),
}
fitted, scores = {}, {}
for name, m in models.items():
    m.fit(Xtr_c, ytr)
    p = m.predict_proba(Xte_c)[:, 1]
    fitted[name], scores[name] = m, p
    results[name] = dict(precision_at_k=precision_at_k(yte, p, K), roc_auc=float(roc_auc_score(yte, p)),
                         pr_auc=float(average_precision_score(yte, p)))
    print(name, {k: round(v, 4) for k, v in results[name].items()})

# ---- Evaluate: P@K for several K (Step 10-11) ----
table = pd.DataFrame(results).T.drop(columns=["ci95"], errors="ignore")
for kk in sorted({max(50, K // 2), K, K * 2}):
    table[f"P@{kk}"] = [precision_at_k(yte, s1, kk) if n.startswith("Level1") else
                        0.0 if n.startswith("Level0") else precision_at_k(yte, scores[n], kk) for n in table.index]
print("\n", table.drop(columns=["note"], errors="ignore").round(4).to_string())
table.to_csv(ROOT / "reports/model_comparison.csv")

# ---- Uncertainty: bootstrap 95% CI of Precision@K (K=300 par numbers noisy hote hain) ----
def boot_ci(y_true, s, k, n=300):
    yv, sv, rng, vals = np.asarray(y_true), np.asarray(s), np.random.default_rng(SEED), []
    for _ in range(n):
        idx = rng.integers(0, len(yv), len(yv))
        vals.append(precision_at_k(yv[idx], sv[idx], k))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
print(f"\n95% bootstrap CI of Precision@{K}:")
results["Level1_speed_limit_group"]["ci95"] = boot_ci(yte, s1, K)
print("  Level1 ", [round(v, 3) for v in results["Level1_speed_limit_group"]["ci95"]])
for n_ in models:
    results[n_]["ci95"] = boot_ci(yte, scores[n_], K)
    print(f"  {n_:21s}", [round(v, 3) for v in results[n_]["ci95"]])

# ---- Choose + save best (Step 12) ----
best = max(models, key=lambda n: (results[n]["precision_at_k"], results[n]["pr_auc"]))
thr = float(np.sort(scores[best])[::-1][K - 1])          # score of the K-th riskiest test call
version = f"{best}-{dt.datetime.now():%Y%m%d%H%M}"
joblib.dump(fitted[best], MODEL_PATH)
json.dump(dict(model_version=version, model=best, features=FEATURES, K=K, threshold=thr,
               metrics=results[best], level1_precision_at_k=results["Level1_speed_limit_group"]["precision_at_k"],
               base_rate=float(y.mean())), open(META_PATH, "w"), indent=2)
print(f"\nBest model: {best} | threshold(High if p>=thr)={thr:.3f} | saved {MODEL_PATH.name}")
print("Beats Level 1?", results[best]["precision_at_k"] > results["Level1_speed_limit_group"]["precision_at_k"])
