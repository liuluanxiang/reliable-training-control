"""从逐运行结果生成案例研究的相关性与性能汇总表。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import spearman_np, pearson_np, area_under_curve, slope_last, mean_std_text
from src.results_io import collect_histories as load_histories, collect_summaries as load_summaries


def collect_histories(results_dir):
    """收集所有逐 epoch 记录，并从目录结构补齐实验键。"""

    frames = []
    for p in Path(results_dir).glob("*/*/seed_*/history.csv"):
        df = pd.read_csv(p)
        if "experiment" not in df.columns:
            df["experiment"] = p.parts[-4]
        if "method" not in df.columns:
            df["method"] = p.parts[-3]
        if "seed" not in df.columns:
            df["seed"] = int(p.parts[-2].replace("seed_", ""))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No history.csv found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def collect_summaries(results_dir):
    """读取主摘要或回退到逐运行摘要。"""

    master = Path(results_dir) / "master_summary.csv"
    if master.exists():
        return pd.read_csv(master)
    rows = []
    for p in Path(results_dir).glob("*/*/seed_*/summary.json"):
        row = json.loads(p.read_text(encoding="utf-8"))
        row.setdefault("experiment", p.parts[-4])
        row.setdefault("method", p.parts[-3])
        row.setdefault("seed", int(p.parts[-2].replace("seed_", "")))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summary files found under {results_dir}")
    return pd.DataFrame(rows)


def table_case1(hist):
    """计算训练阶段信号与泛化指标的相关性表。"""

    required = {"method", "experiment", "dataset", "model", "seed", "pb_signal", "gen_gap_loss", "val_nll", "val_ece"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame()
    rows = []
    df = hist[hist["method"] == "reliability"].copy()
    for keys, d in df.groupby(["experiment", "dataset", "model", "seed"]):
        row = dict(zip(["experiment", "dataset", "model", "seed"], keys))
        row["rho_pb_gengap"] = spearman_np(d["pb_signal"], d["gen_gap_loss"])
        row["rho_pb_valnll"] = spearman_np(d["pb_signal"], d["val_nll"])
        row["rho_pb_ece"] = spearman_np(d["pb_signal"], d["val_ece"])
        row["pearson_pb_gengap"] = pearson_np(d["pb_signal"], d["gen_gap_loss"])
        row["pb_auc"] = area_under_curve(d["pb_signal"].values)
        row["pb_final"] = float(d["pb_signal"].iloc[-1])
        row["pb_last10_slope"] = slope_last(d["pb_signal"].values, 10)
        rows.append(row)
    return pd.DataFrame(rows)


def table_case2(hist):
    """汇总后期退化和控制动作相关案例指标。"""

    required = {"method", "experiment", "dataset", "model", "seed", "pd_signal", "grad_norm", "update_norm", "risk_violation_vt", "val_nll"}
    if hist.empty or not required.issubset(hist.columns):
        return pd.DataFrame()
    rows = []
    df = hist[hist["method"] == "reliability"].copy()
    for keys, d in df.groupby(["experiment", "dataset", "model", "seed"]):
        row = dict(zip(["experiment", "dataset", "model", "seed"], keys))
        row["rho_pd_gradnorm"] = spearman_np(d["pd_signal"], d["grad_norm"])
        row["rho_pd_updatenorm"] = spearman_np(d["pd_signal"], d["update_norm"])
        row["rho_pd_riskviolation"] = spearman_np(d["pd_signal"], d["risk_violation_vt"])
        row["rho_pd_valnll"] = spearman_np(d["pd_signal"], d["val_nll"])
        row["pd_auc"] = area_under_curve(d["pd_signal"].values)
        row["pd_final"] = float(d["pd_signal"].iloc[-1])
        row["grad_norm_final"] = float(d["grad_norm"].iloc[-1])
        row["update_norm_final"] = float(d["update_norm"].iloc[-1])
        th = d["pd_signal"].mean() + 2.0 * d["pd_signal"].std(ddof=0)
        row["pd_spike_count"] = int((d["pd_signal"] > th).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(summary, metrics):
    """按实验与方法聚合指定测试指标。"""

    if summary.empty or not {"experiment", "dataset", "model", "method"}.issubset(summary.columns):
        return pd.DataFrame()
    rows = []
    for keys, d in summary.groupby(["experiment", "dataset", "model", "method"]):
        row = dict(zip(["experiment", "dataset", "model", "method"], keys))
        for m in metrics:
            if m in d.columns:
                row[m] = mean_std_text(d[m])
        rows.append(row)
    return pd.DataFrame(rows)


def table_case5(summary):
    """生成跨设置的最终性能与稳定性案例表。"""

    if summary.empty or not {"experiment", "dataset", "model", "method"}.issubset(summary.columns):
        return pd.DataFrame()
    rows = []
    for keys, d in summary.groupby(["experiment", "dataset", "model", "method"]):
        row = dict(zip(["experiment", "dataset", "model", "method"], keys))
        for m in ["test_acc", "test_nll", "test_ece", "test_brier", "best_epoch", "epochs_run", "late_degradation"]:
            if m in d.columns:
                vals = pd.Series(d[m]).dropna().astype(float)
                row[m + "_mean"] = vals.mean() if len(vals) else np.nan
                row[m + "_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row[m + "_cv"] = vals.std(ddof=1) / vals.mean() if len(vals) > 1 and vals.mean() != 0 else 0.0
        row["n_seeds"] = int(d["seed"].nunique()) if "seed" in d.columns else len(d)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """加载结果、计算案例统计并写出表格文件。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="tables")
    parser.add_argument("--include-quick", action="store_true", help="Include quick-test experiments in outputs.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hist = load_histories(args.results_dir, include_quick=args.include_quick)
    summary = load_summaries(args.results_dir, include_quick=args.include_quick)

    table_case1(hist).to_csv(out / "table_4_3_case1_pac_bayes_signal_validity.csv", index=False)
    table_case2(hist).to_csv(out / "table_4_4_case2_primal_dual_stability.csv", index=False)

    aggregate(summary, [
        "test_acc", "test_top5_acc", "test_nll", "test_ece", "test_brier",
        "final_gen_gap_loss", "late_degradation", "aulc_val_acc",
        "best_epoch", "epochs_run", "lr_reductions", "first_lr_drop_epoch"
    ]).to_csv(out / "table_4_5_case3_lr_scheduling.csv", index=False)

    aggregate(summary, [
        "test_acc", "test_top5_acc", "test_nll", "test_ece", "test_mce",
        "test_brier", "test_aurc", "test_eaurc", "test_confidence_mean",
        "test_entropy_mean", "final_gen_gap_loss", "final_gen_gap_acc",
        "late_degradation"
    ]).to_csv(out / "table_4_6_case4_calibration_generalization.csv", index=False)

    table_case5(summary).to_csv(out / "table_4_7_case5_seed_stability.csv", index=False)

    print(f"Saved case-study tables to {out}")


if __name__ == "__main__":
    main()
