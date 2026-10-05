# Call Severity Triage Assistant (MVP) - GLA University B.Tech project

## Setup (ek baar)
    pip install -r requirements.txt
Dataset pehle se `data/raw/` me hai (STATS19 2023 collisions, 104,258 rows).

## Run order
    python step1_explore.py      # Step 3: dataset explore (SRS Section 3 ke numbers)
    python step2_clean.py        # Steps 4-6: cleaning + target + 5 features
    python step3_train_eval.py   # Steps 7-12: Level0, Level1, ML, Precision@K, best model save
    python test_mvp.py           # Step 16: MVP tests
    streamlit run app.py         # Steps 13-15: prediction pipeline + UI + call logging (calls.db)

Settings: `common.py` me K_DEFAULT (shift capacity), `step2_clean.py` me EXCLUDE_AFTER_SEP17.

## Sprint I (27 Sep) status — MVP done
| SRS ID | What | Where |
|--------|------|-------|
| FR-1.1 | Enter the five call conditions | `app.py` dispatcher form, options from `common.py` |
| FR-1.2 | Reject incomplete entry, naming missing field(s) | `predict.validate()` -> red error in UI, nothing stored |
| FR-2.2 | High risk / Low risk flag | `predict.predict_call()` with threshold from `models/best_model_meta.json` |
| FR-2.4 | Level 0 fallback flag labelled as estimate | `predict_call()` catches model failure -> `Level0-fallback` |
| FR-3.2 | Precision@K for a shift (K = dispatches sent) | `db.shift_precision_at_k()` -> supervisor tab |
| FR-3.3 | Zero-dispatch guard (no divide-by-zero) | message shown instead of a number |
| NFR-1 | Flag within 2 s | measured in `test_mvp.py` (runs in ~1.4 s incl. logging) |
| NFR-3 | Log call ID, inputs, confidence, model version per call | `db.log_call()` -> SQLite `calls.db` (auto-created) |

Model result: HistGradientBoosting, Precision@300 = 0.39 vs Level 1 baseline 0.31 (see `reports/model_comparison.csv`).

UI: dark control-room theme in `theme.py` (glass cards, 3D depth, Inter font) + `.streamlit/config.toml`.
