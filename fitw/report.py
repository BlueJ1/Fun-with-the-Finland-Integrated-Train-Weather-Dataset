"""Build a report and export figures from completed, saved experiments."""

import argparse
import base64
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("tmp/cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spec import PAPER_METRICS

SCENARIOS = ["full", "instant", "categories"]
NAMES = {"full": "1. Full weather", "instant": "2. Instant weather", "categories": "3. Weather categories"}
PROFILES = {
    "chronological": "Chronological, explicit feature IDs and cloud omission",
    "count-consistent": "Chronological, feature counts from Table IV",
    "historical": "Historical-code diagnostic, approximate row order",
    "chronological-author": "Chronological, historical model dependencies",
    "historical-author": "Historical-code diagnostic, historical model dependencies",
}
COLORS = {"full": "#4467b2", "instant": "#da8730", "categories": "#338a6b"}


def plot_curve(root, profile):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    labels = {"r2": "R²", "rmse": "RMSE (minutes)", "mae": "MAE (minutes)", "wmape": "WMAPE (%)"}
    for scenario in SCENARIOS:
        frame = pd.read_csv(root / profile / scenario / "search_budget_curve.csv")
        for ax, (metric, label) in zip(axes.flat, labels.items()):
            ax.plot(frame.search_candidates, frame[metric], marker="o", markersize=3.5,
                    color=COLORS[scenario], label=NAMES[scenario])
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.axhline(PAPER_METRICS[scenario][metric], color=COLORS[scenario], linestyle=":", alpha=0.45)
    for ax in axes[1]:
        ax.set_xlabel("Randomized-search candidates")
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(PROFILES[profile] + "\nSolid: measured holdout metrics; dotted: approximate paper endpoints", fontsize=12)
    dest = root / "figures"
    dest.mkdir(exist_ok=True)
    path = dest / f"{profile}_search_curve.png"
    fig.savefig(path, dpi=180)
    svg = path.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return path


def paired_daily_bootstrap(root, profile="chronological"):
    frames = {s: pd.read_parquet(root / profile / s / "predictions.parquet") for s in SCENARIOS}
    base = frames["categories"].sort_values("row_id").reset_index(drop=True)
    dates = base.event_time.dt.floor("D")
    groups, _ = pd.factorize(dates, sort=True)
    n_groups = int(groups.max() + 1)
    counts = np.bincount(groups, minlength=n_groups)
    rng = np.random.default_rng(42)
    boot_weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=2000)
    denominators = boot_weights @ counts
    output = []
    for other in ["full", "instant"]:
        alternative = frames[other].sort_values("row_id").reset_index(drop=True)
        if not np.array_equal(base.row_id, alternative.row_id):
            raise ValueError("Paired comparisons require identical test events")
        y = base.differenceInMinutes.to_numpy()
        category_error = base.prediction.to_numpy() - y
        other_error = alternative.prediction.to_numpy() - y
        for metric in ["mae", "rmse"]:
            power = 1 if metric == "mae" else 2
            cat_sum = np.bincount(groups, weights=np.abs(category_error) ** power, minlength=n_groups)
            alt_sum = np.bincount(groups, weights=np.abs(other_error) ** power, minlength=n_groups)
            cat_est = (boot_weights @ cat_sum / denominators) ** (1 / power)
            alt_est = (boot_weights @ alt_sum / denominators) ** (1 / power)
            delta = cat_est - alt_est
            observed = np.mean(np.abs(category_error) ** power) ** (1 / power) - np.mean(np.abs(other_error) ** power) ** (1 / power)
            output.append({"comparison": f"categories minus {other}", "metric": metric,
                           "profile": profile,
                           "difference_minutes": float(observed),
                           "daily_block_bootstrap_95pct": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
                           "test_days": n_groups, "replicates": 2000})
    (root / "paired_comparisons.json").write_text(json.dumps(output, indent=2))
    return output


def build(root, data):
    root, data = Path(root), Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    profiles = [p for p in PROFILES if all((root / p / s / "result.json").exists() for s in SCENARIOS)]
    order = ["chronological-author", "historical-author", "chronological", "count-consistent", "historical"]
    profiles.sort(key=order.index)
    if not any(p in profiles for p in ["chronological-author", "chronological"]):
        raise ValueError("All three primary scenarios must finish before reporting")
    rows, collected, pictures = [], {}, {}
    for profile in profiles:
        collected[profile] = {s: json.loads((root / profile / s / "result.json").read_text()) for s in SCENARIOS}
        pictures[profile] = plot_curve(root, profile)
        for scenario, result in collected[profile].items():
            versions = json.loads((root / profile / scenario / "config.json").read_text())["versions"]
            rows.append({"profile": profile, "scenario": scenario,
                         "features": result["feature_count"], **result["test_metrics"],
                         "xgboost_version": versions["xgboost"], "scipy_version": versions["scipy"],
                         "cv_rmse": result["best_candidate"]["mean_cv_rmse"],
                         "best_candidate": result["best_candidate"]["candidate"]})
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    primary_profile = "chronological-author" if "chronological-author" in profiles else "chronological"
    paired = paired_daily_bootstrap(root, primary_profile)
    split = json.loads((root / primary_profile / "full/split.json").read_text())
    primary = collected[primary_profile]
    category_better = primary["categories"]["test_metrics"]["rmse"] < min(primary[s]["test_metrics"]["rmse"] for s in ["full", "instant"])
    verdict = "The reported results were not reproduced under the paper's stated chronological evaluation."
    ranking = ("The category-only model has the lowest measured RMSE, as in the paper."
               if category_better else "The paper's category-only advantage also does not hold in the primary chronological run.")
    header = "# FI-TW replication results\n\n" + verdict + " " + ranking + "\n\n"
    historical_summary = ""
    if "historical-author" in profiles:
        historical_rmse = [collected["historical-author"][s]["test_metrics"]["rmse"] for s in SCENARIOS]
        historical_summary = ("The recovered public-code procedure gives much closer results: RMSE "
                              + ", ".join(f"{value:.2f}" for value in historical_rmse)
                              + " minutes for full weather, instant weather, and weather categories. "
                              "It uses random splitting, shuffled cross-validation, and different tuning choices. "
                              "Exact numerical reproduction and paper-run identity remain unverified.")
        header += historical_summary + "\n\n"
    header += f"The reconstructed Oulu cohort contains **{manifest['rows']:,} events**, matching the reported sample size. Count agreement is not proof of identical row membership or data version. The exact paper-run commit, selected columns, environment, and fitted models are unavailable. A historical dependency export was recovered.\n\n"
    header += f"The main interpretation uses `{primary_profile}`. The author profiles use historical XGBoost 3.0.1, NumPy 2.0.1, pandas 2.2.3, and scikit-learn 1.6.1. Historical SciPy 1.15.1 could not load on this macOS; compatible SciPy 1.17.1 is used instead. Both inclusive and legacy 100-candidate sequences match across the installed modern and author environments; the original SciPy environment could not be run. The complete operating system and Python patch environment are not reproduced.\n\n"
    body = header
    html_parts = [f"<h1>FI-TW replication results</h1><p class='verdict'>{html.escape(verdict)}</p><p>{html.escape(ranking)}</p>",
                  f"<p>101,146 reconstructed Oulu events. Each scenario uses 100 sampled configurations and five validation folds. These are actual locally measured results.</p>",
                  f"<p>The primary interpretation uses <code>{html.escape(primary_profile)}</code>. Author-version profiles use the recovered XGBoost 3.0.1, NumPy 2.0.1, pandas 2.2.3, and scikit-learn 1.6.1 versions. SciPy 1.17.1 replaces the original 1.15.1, whose wheel could not load on this host. Python patch version, operating system, and binary builds also differ from the historical environment.</p>"]
    if historical_summary:
        html_parts.insert(1, f"<p>{html.escape(historical_summary)}</p>")
    for profile in profiles:
        body += "## " + PROFILES[profile] + "\n\n"
        body += "| Scenario | Features | Measured R² | Paper R² | Measured RMSE | Paper RMSE | Measured MAE | Paper MAE | Measured WMAPE | Paper WMAPE |\n"
        body += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        htable = ["<table><thead><tr><th>Scenario</th><th>Features</th><th>R²<br>measured / paper</th><th>RMSE, min<br>measured / paper</th><th>MAE, min<br>measured / paper</th><th>WMAPE<br>measured / paper</th></tr></thead><tbody>"]
        for scenario in SCENARIOS:
            r = collected[profile][scenario]
            m, p = r["test_metrics"], PAPER_METRICS[scenario]
            body += f"| {NAMES[scenario]} | {r['feature_count']} | {m['r2']:.3f} | ~{p['r2']:.2f} | {m['rmse']:.3f} | ~{p['rmse']:.1f} | {m['mae']:.3f} | ~{p['mae']:.1f} | {m['wmape']:.2f}% | ~{p['wmape']:.1f}% |\n"
            htable.append(f"<tr><td>{NAMES[scenario]}</td><td>{r['feature_count']}</td><td>{m['r2']:.3f} / {p['r2']:.2f}</td><td>{m['rmse']:.3f} / {p['rmse']:.1f}</td><td>{m['mae']:.3f} / {p['mae']:.1f}</td><td>{m['wmape']:.2f}% / {p['wmape']:.1f}%</td></tr>")
        htable.append("</tbody></table>")
        body += f"\n![Measured search-budget curves]({pictures[profile].relative_to(root).as_posix()})\n\n"
        html_parts.extend([f"<h2>{PROFILES[profile]}</h2>", "".join(htable)])
        image_data = base64.b64encode(pictures[profile].read_bytes()).decode()
        html_parts.append(f"<img alt='Measured search budget curves' src='data:image/png;base64,{image_data}'>")
        body += "Final models use the best cross-validation score among all 100 candidates. Prefix curves are descriptive evaluations after selection; no test metric chooses hyperparameters or search budget.\n\n"
        if profile.startswith("historical"):
            body += "This diagnostic uses a random 80/20 split, shuffled KFold, MAE selection, legacy weather thresholds, per-month median fills, and reconstructed train-grouped row order. It does not establish the paper's claimed chronological performance. Original filesystem order and interactive feature selections are unrecorded. We report WMAPE rather than the historical code's unstable per-row MAPE, and we do not adopt its test-RMSE search-budget selection.\n\n"
            if profile == "historical-author":
                body += "This profile follows the historical author's raw-weather-only scaling. Operational and category columns retain their original values and order. Earlier archived diagnostics scaled all columns and omitted scale_pos_weight; comparing them with this corrected run combines parameter, scaling, and software changes.\n\n"
                html_parts.append("<p>This diagnostic scales only raw weather predictors, preserving operational and category values. Earlier archived diagnostics scaled all columns and omitted scale_pos_weight. Comparing them with this corrected run combines parameter, scaling, and software changes.</p>")
    body += "## Primary evaluation and preprocessing\n\n"
    body += f"The first {split['development']['rows']:,} chronologically sorted observations form development data. The final {split['test']['rows']:,} form test data, from {split['test']['first_time']} through {split['test']['last_time']}. The development period ends at {split['development']['last_time']}. Five expanding-window validation folds each follow their training fold strictly in time. Every RobustScaler fits only its training fold. XGBoost handles the remaining weather NaNs directly.\n\n"
    body += "The primary feature interpretation follows the eight explicit operational feature IDs in Table IV and the text's cloud omission. This gives 26/16/18 features. The optional count-consistent chronological profile adds day of month and includes cloud, giving the printed 28/18/19; it is not part of the corrected default six-run study. The historical diagnostic uses those printed counts. Gust speed, dew-point temperature, and precipitation amount can contribute to weather categories even though they are not raw model inputs.\n\n"
    body += "Each scenario samples the same 100 configurations with seed 42. The primary search uses inclusive tree counts 100–400 and depths 4–8, learning rates .01/.05/.1, row sampling .7/.8/.9, and feature sampling .7/.8/1.0. The sampled scale_pos_weight values 3.9/4.9/5.9 are passed literally. XGBoost applies these weights to squared-error regression rows whose target equals exactly one minute; this affects 26,908 cohort rows (26.6%) and is not a principled balance of delay classes. Historical integer ranges follow the code's exclusive upper endpoints. The selected model minimizes mean validation RMSE for chronological runs and mean validation MAE for the historical diagnostic. XGBoost uses the CPU histogram tree method. Full details and package versions are recorded in each config.json.\n\n"
    body += "## Cohort and category reconstruction\n\n"
    body += "84 departure-month files cover 2018–2024; 2025 departure files and rolling weather aggregates are excluded. Select station OL and category Long-distance. Of 102,217 source rows, 1,069 have no actual time. Remove those and the two duplicates under the declared retained-feature projection, leaving 101,146. Cancellations with recorded times/targets and negative delays are retained. Overnight events from late December can occur on 1 January 2025. Native data are never modified. All input hashes and the duplicate records are in data/manifest.json.\n\n"
    body += "train_id copies trainNumber. Temporal features follow the pinned historical author convention: actualTime in UTC, fractional hour using minutes, month values 1–12, and weekday Sunday=1 through Saturday=7. These are measurements at the observed station. Actual-time features encode realized delay, so these results cannot establish prediction before departure.\n\n"
    body += "Table II's thresholds apply in printed severity order before any imputation. Missing comparisons fail; Normal/Clear is the residual label, including some incomplete weather. Freezing Rain consumes all Black Ice conditions under the printed hierarchy, so Black Ice is always zero. This is preserved rather than silently fixed. The current author code changes that rule; it is available as an explicit feature-preparation option.\n\n"
    body += "## Metrics and paired scenario comparison\n\n"
    body += "R², RMSE, and MAE use the signed delay target. WMAPE is 100 × sum(abs(y − prediction)) / sum(abs(y)); a signed-denominator alternative is also saved because the paper gives no equation. All paper endpoints are approximate text readings, not exact measurements digitized from Figure 3. The historical code computes a different per-row MAPE, so its percent curve is not an interchangeable reference.\n\n"
    body += "Paired differences use the same primary test rows, with 2,000 daily block bootstrap resamples. Positive differences mean category-only has larger error. Intervals describe this holdout and do not include retraining uncertainty or all long-range temporal dependence.\n\n"
    body += "| Comparison | Metric | Difference, min | Daily bootstrap 95% interval |\n|---|---|---:|---:|\n"
    for row in paired:
        lo, hi = row["daily_block_bootstrap_95pct"]
        body += f"| {row['comparison']} | {row['metric'].upper()} | {row['difference_minutes']:.3f} | [{lo:.3f}, {hi:.3f}] |\n"
    body += "\nTraining-mean, training-median, and zero-delay baseline metrics are saved in every result.json. No railway-only model is reported by the paper; these experiments cannot isolate an incremental benefit from weather.\n\n"
    body += "## Reproduction limits\n\n"
    body += f"The chronological test target has mean {split['test']['target_mean']:.3f} minutes and standard deviation {split['test']['target_std']:.3f} minutes, compared with development mean {split['development']['target_mean']:.3f}. The test period includes 2024, a year with larger observed delays. This is evidence of a distribution change, not a complete explanation of the numerical gap.\n\n"
    historical_profile = next((p for p in ["historical-author", "historical"] if p in profiles), None)
    if historical_profile:
        historical_split = json.loads((root / historical_profile / "full/split.json").read_text())
        hs = historical_split["test"]
        body += f"The source-derived train-grouped random split has test standard deviation {hs['target_std']:.5f} minutes. The reported R² of 0.78 would imply RMSE {hs['target_std'] * np.sqrt(0.22):.3f} minutes on that distribution, consistent with the paper's rounded 8.5. The same reported R² would imply RMSE {split['test']['target_std'] * np.sqrt(0.22):.3f} minutes on this chronological holdout. This supports the historical evaluation as a plausible origin of the scores, without proving exact experiment identity.\n\n"
    body += "The written paper and public source disagree on feature counts, cloud inclusion, split/CV, selection metric, candidate-count interpretation, weather thresholds, and percentage-error metric. The preserved historical commit is an evidence source, not a verified exact paper run. The native archive's network total differs from the paper's approximately 38.5 million records even though the reconstructed Oulu count agrees. The corrected default study uses XGBoost 3.0.1 with the stated SciPy portability exception. Modern-package and alternative feature-count profiles are available as optional runs; their initial parameter-omitted results are archived and excluded. Exact paper-run identity is still unverified. Numerical non-reproduction cannot be attributed to a single discrepancy from these runs.\n\n"
    body += "Diagnostic test curves evaluate multiple search prefixes, after freezing the selection procedure. This departs from the literal 'test used only once' statement, but no test feedback selects or revises any primary model. Sensitivity protocols were declared from source contradictions, not chosen to match reported scores.\n\n"
    body += "Initial experiments mistakenly omitted scale_pos_weight. They were superseded by corrected runs, excluded from these tables and conclusions, and preserved under results/parameter_omission_diagnostics with a warning. Twelve initial scenarios completed and three were interrupted. The correction restores the literal paper/source parameter rather than adjusting it to match the scores.\n\n"
    body += "## Artifacts and rerun\n\n"
    body += "Run `.venv/bin/python run_replication.py` to recreate the complete study. See [README.md](../README.md) for setup and individual commands. Each profile/scenario saves all 100 candidate fold scores, exact split membership, selected parameters, model, scaler, test predictions, feature importances, and curves. [summary.csv](summary.csv) contains the combined results. [paired_comparisons.json](paired_comparisons.json) contains comparison intervals. Figures are exported as PNG and SVG.\n\n"
    body += "Sources: [supplied paper](https://arxiv.org/abs/2609.11277), [author dataset](https://www.kaggle.com/datasets/viniborin/finland-integrated-train-weather-dataset-fi-tw), [pinned historical source](https://github.com/borinvini/Railway-FMI-Data_training/tree/c28b188948fe42ccc0da2c0f84a81405cc1e2a72). Local evidence is in [protocol_audit.md](../references/protocol_audit.md), [source_search.md](../references/source_search.md), and [data_audit.md](../references/data_audit.md).\n"
    (root / "REPORT.md").write_text(body)
    html_parts.extend(["<h2>Method and limits</h2>",
        "<p>The primary run follows chronological 80/20 evaluation and fold-fitted preprocessing. Its 26/16/18 predictors follow the explicit feature IDs and cloud omission; the historical diagnostic uses the alternative printed counts 28/18/19. The historical-code diagnostic uses random splitting and cannot validate chronological generalization.</p>",
        "<p>The historical diagnostics reconstruct train-number grouping within each sorted source month. The original unsorted filesystem enumeration and interactive feature selections were not preserved, so the original random partition is not verified.</p>",
        "<p>The Oulu sample count matches, but exact author row membership and experiment versions are unavailable. Temporal predictors derive from actualTime, so this is event-time reconstruction, not verified advance prediction. Printed Black Ice rules are unreachable. Paper percentages and historical code MAPE differ.</p>",
        "<p>All sampled parameters are passed literally. In XGBoost, scale_pos_weight reweights squared-error regression observations whose delay equals exactly one minute, affecting 26.6% of this cohort. Initial runs mistakenly omitted it; those superseded diagnostics are archived separately and excluded from this report.</p>",
        "<p>Curves show descriptive holdout evaluations at predeclared search budgets; model selection uses only validation scores. WMAPE uses the sum of absolute actual delays. See the accompanying REPORT.md for complete evidence and assumptions.</p>"])
    style = "body{font:16px/1.6 system-ui,sans-serif;color:#18212f;background:#f6f8fb;margin:0}main{max-width:1120px;margin:auto;padding:40px 28px;background:white}h1{font-size:34px;line-height:1.2}h2{margin-top:44px;font-size:22px}p{max-width:960px}.verdict{padding:16px 20px;background:#fff1df;border-left:4px solid #be7925;font-weight:600}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:right;padding:12px;border-bottom:1px solid #dde3ec}th:first-child,td:first-child{text-align:left}th{background:#f2f5f9}img{width:100%;height:auto;margin:20px 0}@media(max-width:700px){main{padding:22px 12px}table{font-size:11px}th,td{padding:8px 4px}}"
    (root / "replication.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>FI-TW replication results</title><style>" + style + "</style><main>" + "".join(html_parts) + "</main></html>")
    print(f"Wrote {root / 'REPORT.md'}, summary.csv, replication.html, and figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    build(args.results, args.data)


if __name__ == "__main__":
    main()
