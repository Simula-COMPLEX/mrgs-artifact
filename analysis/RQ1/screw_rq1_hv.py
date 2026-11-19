#!/usr/bin/env python3
"""Screw detection RQ1: final-HV Wilcoxon tests and HV convergence.

Reads per-image summary.json and convergence.csv files.
Does not modify raw results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

MODELS = ("origSCDM", "noscrewSCDM", "fixtureSCDM") # presented as SCDM1, SCDM2, and SCDM3 in the paper.
ALGORITHMS = ("Random Search", "NSGA-II", "SPEA2", "MOEA/D")
ROOT = Path.home() / "D1" / "face_detection" / "screw_detection"
DIRECTORIES = {
    "origSCDM": {
        "Random Search": "initial_dataset/random_jaccard_screw_orig/all_train",
        "NSGA-II": "initial_dataset/nsga2_jaccard_screw_orig/all_train",
        "SPEA2": "initial_dataset/spea2_jaccard_screw_orig/all_train",
        "MOEA/D": "initial_dataset/moead_jaccard_screw_orig/all_train",
    },
    "noscrewSCDM": {
        "Random Search": "added_more_noscrews_at_diff_exposure/random_jaccard_noscrews/all_train",
        "NSGA-II": "added_more_noscrews_at_diff_exposure/nsga2_jaccard_noscrews/all_train",
        "SPEA2": "added_more_noscrews_at_diff_exposure/spea2_jaccard_noscrews/all_train",
        "MOEA/D": "added_more_noscrews_at_diff_exposure/moead_jaccard_noscrews/all_train",
    },
    "fixtureSCDM": {
        "Random Search": "added_screw_fixture/random_jaccard_fixture/all_train",
        "NSGA-II": "added_screw_fixture/nsga2_jaccard_fixture/all_train",
        "SPEA2": "added_screw_fixture/spea2_jaccard_fixture/all_train",
        "MOEA/D": "added_screw_fixture/moead_jaccard_fixture/all_train",
    },
}


def detect_algorithm(method):
    name = str(method).lower()
    for token, algorithm in (("random", "Random Search"), ("nsga", "NSGA-II"),
                             ("spea", "SPEA2"), ("moead", "MOEA/D"),
                             ("moea/d", "MOEA/D")):
        if token in name:
            return algorithm
    return None


def load_results(root, runs):
    records = []
    for model in MODELS:
        for algorithm in ALGORITHMS:
            directory = root / DIRECTORIES[model][algorithm]
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing directory: {directory}")
            for run in range(1, runs + 1):
                run_dir = directory / f"run_{run:02d}"
                if not run_dir.is_dir():
                    print(f"[warning] Missing run: {run_dir}")
                    continue
                summaries = sorted(run_dir.glob("*/summary.json"))
                if not summaries:
                    print(f"[warning] No image summaries in: {run_dir}")
                for path in summaries:
                    try:
                        with path.open(encoding="utf-8") as file:
                            summary = json.load(file)
                        method = summary.get("method", "")
                        detected = detect_algorithm(method)
                        if method and detected != algorithm:
                            raise ValueError(f"Algorithm mismatch: expected {algorithm}, found {method!r}: {path}")
                        image = str(summary.get("image_name", "")).strip()
                        hv = float(summary["final_hv"])
                        if not image or not np.isfinite(hv) or not (0 <= hv <= 1.21 + 1e-6):
                            raise ValueError("Missing image name or invalid final HV")
                    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        if "Algorithm mismatch" in str(exc):
                            raise
                        print(f"[warning] Skipping {path}: {exc}")
                        continue
                    records.append((model, algorithm, run, image, hv, str(path)))
    if not records:
        raise ValueError("No valid per-image summary.json files found")
    data = pd.DataFrame(records, columns=["model", "algorithm", "run", "image_name", "final_hv", "summary_path"])
    if data.duplicated(["model", "algorithm", "run", "image_name"]).any():
        raise ValueError("Duplicate image/run/algorithm records")
    return data



def select_complete(df, runs):
    keys = ["model", "image_name"]
    counts = df.groupby(keys + ["algorithm"])["run"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=ALGORITHMS, fill_value=0)
    complete = counts.eq(runs).all(axis=1)
    for model in MODELS:
        sub = complete[complete.index.get_level_values("model") == model]
        print(f"{model}: {int(sub.sum())}/{len(sub)} fully paired images")
        if not sub.any():
            raise ValueError(f"No fully paired images for {model}")
    return df.set_index(keys).loc[counts.index[complete]].reset_index()

IMAGE_KEYS = ["image_name"]
DEFAULT_ROOT = ROOT
DEFAULT_OUTPUT = "screw_rq1_results"
FIGURE_PREFIX = "rq1_screw_hv_convergence"
FIGURE_TITLE = "Screw detection: HV convergence by model"

def rank_biserial(diff):
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def analyze_final_hv(paired, image_keys):
    means = paired.groupby(["model"] + image_keys + ["algorithm"],
                           as_index=False)["final_hv"].mean()
    rows = []
    for model in MODELS:
        wide = means[means.model == model].pivot(
            index=image_keys, columns="algorithm", values="final_hv")
        baseline = wide["Random Search"].to_numpy()
        for algorithm in ALGORITHMS[1:]:
            candidate = wide[algorithm].to_numpy()
            diff = candidate - baseline
            p = 1.0 if np.all(diff == 0) else float(
                wilcoxon(diff, alternative="two-sided").pvalue)
            rows.append(dict(
                model=model, algorithm=algorithm, baseline="Random Search",
                n_images=len(diff), mean_hv=float(candidate.mean()),
                mean_random_hv=float(baseline.mean()),
                mean_delta_hv=float(diff.mean()),
                median_delta_hv=float(np.median(diff)),
                wilcoxon_p=p, rank_biserial=rank_biserial(diff),
                fraction_images_better=float(np.mean(diff > 0))))
    tests = pd.DataFrame(rows)
    tests["holm_p"] = holm_adjust(tests.wilcoxon_p.to_numpy())
    tests["significantly_better"] = (
        (tests.holm_p < 0.05) & (tests.mean_delta_hv > 0))
    return tests


def read_curve(summary_path, final_hv, grid):
    path = Path(summary_path).with_name("convergence.csv")
    c = pd.read_csv(path, usecols=["n_eval", "hv"])
    c["n_eval"] = pd.to_numeric(c["n_eval"], errors="coerce")
    c["hv"] = pd.to_numeric(c["hv"], errors="coerce")
    c = c.replace([np.inf, -np.inf], np.nan).dropna()
    c = c.sort_values("n_eval").drop_duplicates("n_eval", keep="last")
    if len(c) < 2:
        raise ValueError("fewer than two valid convergence points")
    x, y = c.n_eval.to_numpy(), c.hv.to_numpy()
    if abs(y[-1] - final_hv) > 1e-6:
        raise ValueError(
            f"convergence endpoint {y[-1]:.8f} != final_hv {final_hv:.8f}")
    if np.any(np.diff(x) <= 0):
        raise ValueError("non-increasing evaluation counts")
    # No extrapolation beyond observed evaluations.
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def convergence_summary(paired, image_keys, runs, grid, out):
    # Keep the exact same image set across all four algorithms, and
    # require all runs to have valid convergence curves.
    per_image = {}
    issues = []
    grouping = ["model"] + image_keys + ["algorithm"]
    for key, group in paired.groupby(grouping):
        curves = []
        for record in group.itertuples(index=False):
            try:
                curves.append(read_curve(
                    record.summary_path, record.final_hv, grid))
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                issues.append(f"{record.summary_path}: {exc}")
        if len(curves) == runs:
            stack = np.vstack(curves)
            # Require coverage at all common grid points.
            if np.isfinite(stack).all():
                per_image[key] = stack.mean(axis=0)
            else:
                issues.append(f"{key}: incomplete evaluation-grid coverage")
        else:
            issues.append(f"{key}: {len(curves)}/{runs} usable curves")

    if issues:
        (out / "convergence_warnings.txt").write_text(
            "\n".join(issues) + "\n", encoding="utf-8")
        print(f"[warning] {len(issues)} convergence issues; see convergence_warnings.txt")

    by_model = {}
    for model in MODELS:
        image_ids = {
            key[1:-1] for key in per_image if key[0] == model
        }
        complete = sorted(image_id for image_id in image_ids if all(
            (model, *image_id, algorithm) in per_image
            for algorithm in ALGORITHMS))
        if not complete:
            raise ValueError(f"No fully paired convergence curves for {model}")
        by_model[model] = {
            algorithm: np.vstack([
                per_image[(model, *image_id, algorithm)]
                for image_id in complete])
            for algorithm in ALGORITHMS
        }
        print(f"{model}: {len(complete)} images with complete convergence curves")

    rows = []
    for model, algorithms in by_model.items():
        for algorithm, stack in algorithms.items():
            n = len(stack)
            means = stack.mean(axis=0)
            sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(grid))
            for budget, mean, err in zip(grid, means, sem):
                rows.append(dict(
                    model=model, algorithm=algorithm, n_images=n,
                    n_eval=float(budget), mean_hv=float(mean),
                    ci95_lower=max(0.0, float(mean - 1.96 * err)),
                    ci95_upper=min(1.21, float(mean + 1.96 * err))))
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "rq1_convergence_summary.csv", index=False)
    return curves, by_model


def plot_convergence(curves, by_model, tests, out, prefix, title):
    nmodels = len(MODELS)
    fig, axes = plt.subplots(
        1, nmodels, figsize=(6.0 * nmodels, 5.5),
        sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    colors = {
        "Random Search": "#1f77b4", "NSGA-II": "#ff7f0e",
        "SPEA2": "#2ca02c", "MOEA/D": "#d62728"}
    labels = {"Random Search": "RS", "NSGA-II": "NSGA-II",
              "SPEA2": "SPEA2", "MOEA/D": "MOEA/D"}
    for ax, model in zip(axes, MODELS):
        for algorithm in ALGORITHMS:
            sub = curves[(curves.model == model) &
                         (curves.algorithm == algorithm)].sort_values("n_eval")
            ax.plot(sub.n_eval, sub.mean_hv, label=labels[algorithm],
                    color=colors[algorithm], lw=2)
            ax.fill_between(sub.n_eval.to_numpy(),
                            sub.ci95_lower.to_numpy(),
                            sub.ci95_upper.to_numpy(),
                            color=colors[algorithm], alpha=.12)
        n = len(by_model[model]["Random Search"])
        ax.set_title(f"{model}  ·  {n} paired images")
        ax.set_xlabel("Candidate evaluations")
        ax.grid(alpha=.2)
        comparisons = tests[tests.model == model]
        annotation = ["Final-HV tests vs RS (Holm-adjusted):"]
        for r in comparisons.itertuples():
            direction = "+" if r.mean_delta_hv > 0 else "−" if r.mean_delta_hv < 0 else "="
            annotation.append(
                f"{labels[r.algorithm]}: Δ={r.mean_delta_hv:+.3f}, "
                f"p={r.holm_p:.2g} ({direction})")
        ax.text(.98, .04, "\n".join(annotation), transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=.92,
                          boxstyle="round,pad=.5"))
    axes[0].set_ylabel("Mean hypervolume (approx. 95% CI across images)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4)
    fig.suptitle(title, fontsize=14)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{prefix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=10000)
    parser.add_argument("--points", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 2 or args.max_evals <= 40 or args.points < 2:
        parser.error("Require --runs >= 2, --max-evals > 40, --points >= 2")
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset root not found: {root}")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = load_results(root, args.runs)
    paired = select_complete(raw, args.runs)
    tests = analyze_final_hv(paired, IMAGE_KEYS)
    tests.to_csv(out / "rq1_statistical_results.csv",
                 index=False, float_format="%.8g")
    grid = np.linspace(40, args.max_evals, args.points)
    curves, by_model = convergence_summary(
        paired, IMAGE_KEYS, args.runs, grid, out)
    plot_convergence(curves, by_model, tests, out, FIGURE_PREFIX, FIGURE_TITLE)
    print("\n" + tests.to_string(index=False))
    print(f"\nResults saved in {out}")


if __name__ == "__main__":
    main()
