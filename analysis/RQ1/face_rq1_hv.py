#!/usr/bin/env python3
"""Face detection RQ1: final-HV Wilcoxon tests and HV convergence, with densities pooled.

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
from scipy.stats import wilcoxon, rankdata

MODELS = ("2023",) # presented as FDM in the paper.
DENSITIES = ("low_density", "medium_density", "high_density")
ALGORITHMS = ("Random Search", "NSGA-II", "SPEA2", "MOEA/D")
DATASET_ROOT = Path.home() / "D1" / "face_detection" / "WIDER_val_density_600"

RESULT_DIRECTORIES = {
    ("2023", "low_density", "Random Search"):
        "random_yunet_jaccard_2023_low/low_density_final",
    ("2023", "low_density", "NSGA-II"):
        "nsga_yunet_jaccard_2023_low/low_density_final",
    ("2023", "low_density", "SPEA2"):
        "spea2_yunet_jaccard_2023_low/low_density_final",
    ("2023", "low_density", "MOEA/D"):
        "moead_yunet_jaccard_2023_low/low_density_final",

    ("2023", "medium_density", "Random Search"):
        "random_yunet_jaccard_2023_medium/medium_density_final",
    ("2023", "medium_density", "NSGA-II"):
        "nsga_yunet_jaccard_2023_medium/medium_density_final",
    ("2023", "medium_density", "SPEA2"):
        "spea2_yunet_jaccard_2023_medium/medium_density_final",
    ("2023", "medium_density", "MOEA/D"):
        "moead_yunet_jaccard_2023_medium/medium_density_final",

    ("2023", "high_density", "Random Search"):
        "random_yunet_jaccard_2023_high/high_density_final",
    ("2023", "high_density", "NSGA-II"):
        "nsga_yunet_jaccard_2023_high/high_density_final",
    ("2023", "high_density", "SPEA2"):
        "spea2_yunet_jaccard_2023_high/high_density_final",
    ("2023", "high_density", "MOEA/D"):
        "moead_yunet_jaccard_2023_high/high_density_final",


}

def algorithm_from_method(method):
    value = method.lower()
    for fragment, name in (("random", "Random Search"), ("nsga", "NSGA-II"),
                           ("spea", "SPEA2"), ("moead", "MOEA/D"), ("moea/d", "MOEA/D")):
        if fragment in value:
            return name
    return None


def load_results(root, runs):
    records = []
    expected = {(m, d, a) for m in MODELS for d in DENSITIES for a in ALGORITHMS}
    if set(RESULT_DIRECTORIES) != expected:
        raise ValueError("RESULT_DIRECTORIES must contain exactly 12 model/density/algorithm keys")

    for (model, density, algorithm), relative in RESULT_DIRECTORIES.items():
        directory = root / relative
        if not directory.is_dir() and directory.name == density + "_final":
            directory = directory.with_name(density)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing experiment directory: {directory}")

        for run in range(1, runs + 1):
            run_dir = directory / f"run_{run:02d}"
            if not run_dir.is_dir():
                print(f"[warning] Missing run: {run_dir}")
                continue
            for path in sorted(run_dir.glob("*/summary.json")):
                try:
                    with path.open(encoding="utf-8") as f:
                        summary = json.load(f)
                    method = str(summary.get("method", ""))
                    if method and algorithm_from_method(method) != algorithm:
                        raise ValueError(f"Algorithm mismatch in {path}: {method}")
                    if algorithm == "Random Search" and method != "Population-limited Random Search":
                        raise ValueError(f"Random Search is not population-limited: {path}")
                    category = str(summary.get("category", ""))
                    if category and not category.startswith(density):
                        raise ValueError(f"Density mismatch in {path}: {category}")
                    model_path = str(summary.get("model_path", ""))
                    if ("2023" in model_path or "2026" in model_path) and model not in model_path:
                        raise ValueError(f"YuNet version mismatch in {path}: {model_path}")
                    if summary.get("population") is not None and int(summary["population"]) != 40:
                        raise ValueError(f"Unexpected population in {path}")
                    if summary.get("generations") is not None and int(summary["generations"]) != 250:
                        raise ValueError(f"Unexpected generations in {path}")
                    image = str(summary.get("image_name", ""))
                    hv = float(summary["final_hv"])
                    if not image or not np.isfinite(hv) or not 0 <= hv <= 1.21 + 1e-6:
                        raise ValueError("Missing image name or invalid final HV")
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    # Wrong experiment configuration is dangerous: do not silently include it.
                    if isinstance(exc, ValueError) and ("mismatch" in str(exc) or
                            "Unexpected" in str(exc) or "not population-limited" in str(exc)):
                        raise
                    print(f"[warning] Skipping {path}: {exc}")
                    continue
                records.append((model, density, algorithm, run, image, hv, str(path)))

    if not records:
        raise ValueError("No valid summary.json results found")
    df = pd.DataFrame(records, columns=["model", "density", "algorithm", "run", "image_name", "final_hv", "summary_path"])
    key = ["model", "density", "algorithm", "run", "image_name"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate records for the same image, run and algorithm")
    return df



def select_complete(df, runs):
    keys = ["model", "density", "image_name"]
    counts = df.groupby(keys + ["algorithm"])["run"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=ALGORITHMS, fill_value=0)
    complete = counts.eq(runs).all(axis=1)
    for model in MODELS:
        sub = complete[complete.index.get_level_values("model") == model]
        print(f"YuNet {model}: {int(sub.sum())}/{len(sub)} fully paired images")
        if not sub.any():
            raise ValueError(f"No fully paired images for YuNet {model}")
    return df.set_index(keys).loc[counts.index[complete]].reset_index()

IMAGE_KEYS = ["density", "image_name"]
DEFAULT_ROOT = DATASET_ROOT
DEFAULT_OUTPUT = "pal_rq1_results_simple"
FIGURE_PREFIX = "rq1_yunet_hv_convergence"
FIGURE_TITLE = "YuNet: Hypervolume convergence"

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


def plot_convergence(curves, by_model, out, prefix, title):
    """Single clean YuNet panel; show final mean HV in the legend."""
    model = MODELS[0]
    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")
    colors = {
        "Random Search": "#1f77b4", "NSGA-II": "#ff7f0e",
        "SPEA2": "#2ca02c", "MOEA/D": "#d62728"}
    labels = {"Random Search": "RS", "NSGA-II": "NSGA-II",
              "SPEA2": "SPEA2", "MOEA/D": "MOEA/D"}

    for algorithm in ALGORITHMS:
        sub = curves[(curves.model == model) &
                     (curves.algorithm == algorithm)].sort_values("n_eval")
        final_mean = float(sub.mean_hv.iloc[-1])
        ax.plot(sub.n_eval, sub.mean_hv,
                label=f"{labels[algorithm]} (HV={final_mean:.3f})",
                color=colors[algorithm], linewidth=2.2)
        ax.fill_between(sub.n_eval.to_numpy(),
                        sub.ci95_lower.to_numpy(), sub.ci95_upper.to_numpy(),
                        color=colors[algorithm], alpha=0.12)

    n = len(by_model[model]["Random Search"])
    ax.set_title(f"YuNet · {n} paired images (all densities)", fontsize=12)
    ax.set_xlabel("Candidate evaluations")
    ax.set_ylabel("Mean hypervolume (approx. 95% CI across images)")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
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
    plot_convergence(curves, by_model, out, FIGURE_PREFIX, FIGURE_TITLE)
    print("\n" + tests.to_string(index=False))
    print(f"\nResults saved in {out}")


if __name__ == "__main__":
    main()
