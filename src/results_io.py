import json
from pathlib import Path

import pandas as pd


def infer_experiment_type(experiment_name):
    name = str(experiment_name or "").lower()
    if name.startswith("quick"):
        return "quick"
    if name.startswith("main"):
        return "main"
    if name.startswith("supp"):
        return "supplement"
    if name.startswith("ablation"):
        return "ablation"
    if name.startswith("sens") or name.startswith("sensitivity"):
        return "sensitivity"
    return "unspecified"


def ensure_experiment_type(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    if "experiment_type" not in out.columns:
        out["experiment_type"] = None
    if "experiment" in out.columns:
        missing = out["experiment_type"].isna() | (out["experiment_type"].astype(str).str.strip() == "")
        out.loc[missing, "experiment_type"] = out.loc[missing, "experiment"].map(infer_experiment_type)
    else:
        out["experiment_type"] = out["experiment_type"].fillna("unspecified")
    return out


def filter_quick(df, include_quick=False):
    out = ensure_experiment_type(df)
    if out is None or out.empty or include_quick:
        return out
    return out[out["experiment_type"] != "quick"].copy()


def collect_histories(results_dir, include_quick=False):
    frames = []
    for path in Path(results_dir).glob("*/*/seed_*/history.csv"):
        try:
            df = pd.read_csv(path)
            if "experiment" not in df.columns:
                df["experiment"] = path.parts[-4]
            if "method" not in df.columns:
                df["method"] = path.parts[-3]
            if "seed" not in df.columns:
                df["seed"] = int(path.parts[-2].replace("seed_", ""))
            if "experiment_type" not in df.columns:
                df["experiment_type"] = infer_experiment_type(df["experiment"].iloc[0])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return filter_quick(pd.concat(frames, ignore_index=True), include_quick=include_quick)


def collect_summaries(results_dir, include_quick=False):
    results_dir = Path(results_dir)
    master = results_dir / "master_summary.csv"
    if master.exists():
        try:
            return filter_quick(pd.read_csv(master), include_quick=include_quick)
        except Exception:
            pass

    rows = []
    for path in results_dir.glob("*/*/seed_*/summary.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row.setdefault("experiment", path.parts[-4])
            row.setdefault("method", path.parts[-3])
            row.setdefault("seed", int(path.parts[-2].replace("seed_", "")))
            row.setdefault("experiment_type", infer_experiment_type(row.get("experiment")))
            rows.append(row)
        except Exception:
            continue
    return filter_quick(pd.DataFrame(rows), include_quick=include_quick)
