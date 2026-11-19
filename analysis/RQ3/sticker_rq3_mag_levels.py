#!/usr/bin/env python3
"""RQ3 Sticker Detection: best MR violation severity in shared magnitude tertiles.

Analyzes SSD300 (STDM1) and SSDLite320 (STDM2) INDEPENDENTLY.
Uses runs 02--11, all saved candidates per image/run, shared Q33/Q67
within each model, and paired image/run/level observations across all MOEAs.
For each image, averages per-run best severity across matched runs; paired
Wilcoxon tests operate on image-level values with Holm correction per level.
Prints FDM-style summary tables and tests. Saves one PDF boxplot per model.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGS = ('NSGA-II', 'SPEA2', 'MOEA/D')
LEVELS = ('Low', 'Moderate', 'High')
MODELS = {
    'ssd300_vgg16': 'STDM1',
    'ssdlite320_mobilenet_v3_large': 'STDM2',
}
ALGORITHM_FOLDERS = {
    'NSGA-II': 'nsga2_jaccard_sticker',
    'SPEA2': 'spea2_jaccard_sticker',
    'MOEA/D': 'moead_jaccard_sticker',
}


def find_run_dir(root, model, algorithm, run):
    """Support both model/algorithm/valid/run_XX and algorithm/model/valid/run_XX."""
    folder = ALGORITHM_FOLDERS[algorithm]
    r = f'run_{run:02d}'
    options = [
        root / model / folder / 'valid' / r,
        root / model / folder / r,
        root / folder / model / 'valid' / r,
        root / folder / model / r,
        root / model / 'valid' / folder / r,
    ]
    found = [p for p in options if p.is_dir()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f'Ambiguous run directory for {model}, {algorithm}, {r}: {found}')
    # Fallback: exact directory components, not substring matching.
    matches = [p for p in root.rglob(r) if p.is_dir()
               and model in p.parts and folder in p.parts and 'valid' in p.parts]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous run directories for {model}, {algorithm}, {r}: {matches[:5]}')
    return None


def read_model(root, model, run_start, runs, expected, strict):
    chunks, warnings = [], []
    for alg in ALGS:
        for run in range(run_start, run_start + runs):
            run_dir = find_run_dir(root, model, alg, run)
            if run_dir is None:
                warnings.append(f'Missing run: {model} / {alg} / run_{run:02d}')
                continue
            paths = sorted(p for p in run_dir.rglob('saved_candidates.csv')
                           if p.parent.name == 'ALL_SAVED')
            if not paths:
                warnings.append(f'No ALL_SAVED/saved_candidates.csv under {run_dir}')
            for path in paths:
                try:
                    frame = pd.read_csv(path)
                except Exception as exc:
                    warnings.append(f'Could not read {path}: {exc}')
                    continue
                needed = {'image_name', 'mr_score', 'perturbation_magnitude'}
                if not needed.issubset(frame.columns):
                    warnings.append(f'Missing {needed-set(frame.columns)} in {path}')
                    continue
                frame = frame[['image_name', 'mr_score', 'perturbation_magnitude']].copy()
                frame['mr_score'] = pd.to_numeric(frame['mr_score'], errors='coerce')
                frame['perturbation_magnitude'] = pd.to_numeric(frame['perturbation_magnitude'], errors='coerce')
                n0 = len(frame)
                frame = frame.dropna(subset=['image_name', 'mr_score', 'perturbation_magnitude'])
                frame = frame.loc[frame.mr_score.between(0, 1) & frame.perturbation_magnitude.between(0, 1)]
                if len(frame) != n0:
                    warnings.append(f'Excluded {n0-len(frame)} invalid rows in {path}')
                frame['image_name'] = frame.image_name.astype(str)
                frame['algorithm'] = alg
                frame['run'] = run
                chunks.append(frame)
    if not chunks:
        raise RuntimeError(f'No saved candidates found for {model} under {root}')
    df = pd.concat(chunks, ignore_index=True)
    keys = ['image_name', 'run', 'algorithm']
    counts = df.groupby(keys).size()
    bad = counts[counts != expected]
    if len(bad):
        message = f'{len(bad)} image/run/algorithm units contain != {expected} candidates'
        if strict:
            raise RuntimeError(f'{model}: {message}. First examples: {bad.head().to_dict()}')
        warnings.append(message)
    # Match the complete image/run units across algorithms before estimating thresholds.
    n_algs = df.groupby(['image_name', 'run']).algorithm.nunique()
    common = n_algs[n_algs.eq(len(ALGS))].index
    if not len(common):
        raise RuntimeError(f'No common image/run units for {model}; inspect paths/image_name fields')
    before = len(df)
    df = df.set_index(['image_name', 'run']).loc[common].reset_index()
    if len(df) != before:
        warnings.append(f'Paired filtering removed {before-len(df)} candidate rows')
    return df, warnings


def shared_tertiles(df):
    """Balanced pooled weighted empirical quantiles (same as FDM script).

    Each algorithm gets equal weight; each image/run unit within an algorithm
    gets equal weight; each saved candidate within a unit gets equal weight.
    """
    keys = ['algorithm', 'image_name', 'run']
    n_in_unit = df.groupby(keys).mr_score.transform('size').to_numpy()
    n_units = df[keys].drop_duplicates().groupby('algorithm').size().to_dict()
    weights = (1.0 / n_in_unit) / df.algorithm.map(n_units).to_numpy()
    weights /= len(n_units)
    vals = df.perturbation_magnitude.to_numpy()
    order = np.argsort(vals, kind='stable')
    v, w = vals[order], weights[order]
    cumulative = np.cumsum(w) / w.sum()
    return tuple(float(v[min(np.searchsorted(cumulative, q), len(v)-1)])
                 for q in (1/3, 2/3))


def analyze(df, q33, q67):
    df = df.copy()
    df['level'] = np.select(
        [df.perturbation_magnitude <= q33, df.perturbation_magnitude <= q67],
        ['Low', 'Moderate'], default='High')
    total_units = df[['image_name', 'run']].drop_duplicates().shape[0]
    per_run = (df.groupby(['image_name', 'run', 'algorithm', 'level'], as_index=False)
               .agg(best_mr=('mr_score', 'max'), n_candidates=('mr_score', 'size')))
    # Only image/run/level cells where ALL THREE algorithms have >=1 candidate.
    keys = ['image_name', 'run', 'level']
    n_algs = per_run.groupby(keys).algorithm.nunique()
    common = n_algs[n_algs.eq(len(ALGS))].index
    matched = (per_run.set_index(keys).loc[common].reset_index()
               if len(common) else per_run.iloc[:0].copy())
    # Paired image-level unit: average over exactly the same matched runs.
    image = (matched.groupby(['image_name', 'algorithm', 'level'], as_index=False)
             .agg(best_mr=('best_mr', 'mean'), n_matched_runs=('run', 'nunique')))
    rows = []
    for level in LEVELS:
        n_matched_units = int(sum(n_algs.eq(len(ALGS)) & (n_algs.index.get_level_values('level') == level)))
        for alg in ALGS:
            raw = df[(df.algorithm == alg) & (df.level == level)]
            avail = per_run[(per_run.algorithm == alg) & (per_run.level == level)]
            img = image[(image.algorithm == alg) & (image.level == level)]
            vals = img.best_mr.to_numpy()
            rows.append({
                'level': level, 'algorithm': alg,
                'candidates': len(raw),
                'candidate_pct': 100 * len(raw) / max(1, sum(df.algorithm == alg)),
                'individual_coverage_pct': 100 * len(avail) / total_units,
                'matched_units': n_matched_units,
                'matched_coverage_pct': 100 * n_matched_units / total_units,
                'images': len(img),
                'mean_best': np.mean(vals) if len(vals) else np.nan,
                'sd_best': np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                'median_best': np.median(vals) if len(vals) else np.nan,
                'q25_best': np.percentile(vals, 25) if len(vals) else np.nan,
                'q75_best': np.percentile(vals, 75) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows), image, total_units


def rank_biserial(diffs):
    """Paired rank-biserial correlation using the Wilcoxon zero_method='wilcox' convention."""
    nonzero = diffs[~np.isclose(diffs, 0, atol=1e-12)]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero), method='average')
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def effect_size_label(rrb):
    """Magnitude categories based on absolute rank-biserial correlation."""
    magnitude = abs(rrb)
    if magnitude < 0.10:
        return 'Negligible'
    if magnitude < 0.30:
        return 'Small'
    if magnitude < 0.50:
        return 'Medium'
    return 'Large'


def tests(image):
    pairs = [('NSGA-II', 'SPEA2'), ('NSGA-II', 'MOEA/D'), ('SPEA2', 'MOEA/D')]
    rows = []
    for level in LEVELS:
        wide = image[image.level == level].pivot(index='image_name', columns='algorithm', values='best_mr')
        batch = []
        for a, b in pairs:
            paired = wide.reindex(columns=[a, b]).dropna()
            diffs = (paired[a] - paired[b]).to_numpy()
            nonzero = diffs[~np.isclose(diffs, 0, atol=1e-12)]
            p = float(wilcoxon(nonzero, alternative='two-sided').pvalue) if len(nonzero) else 1.0
            rrb = rank_biserial(diffs)
            batch.append({'level': level, 'comparison': f'{a} vs {b}',
                          'n_images': len(diffs), 'mean_difference': np.mean(diffs) if len(diffs) else np.nan,
                          'p_raw': p, 'r_rb': rrb, 'abs_r_rb': abs(rrb),
                          'effect_size': effect_size_label(rrb)})
        ordered = np.argsort([row['p_raw'] for row in batch])
        adjusted = 0.0
        for rank, idx in enumerate(ordered):
            adjusted = max(adjusted, (len(batch) - rank) * batch[idx]['p_raw'])
            batch[idx]['p_holm'] = min(1.0, adjusted)
        rows.extend(batch)
    return pd.DataFrame(rows)


def save_boxplot(image, model_label, outdir):
    """One publication-ready PDF per model, using matched image-level observations."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions, datasets, labels = [], [], []
    offsets = [-0.25, 0, 0.25]
    for i, level in enumerate(LEVELS):
        for j, alg in enumerate(ALGS):
            values = image.loc[(image.level == level) & (image.algorithm == alg), 'best_mr'].dropna().to_numpy()
            if len(values):
                positions.append(i + 1 + offsets[j])
                datasets.append(values)
                labels.append((i, j))
    if not datasets:
        plt.close(fig)
        print(f'WARNING: no observations for {model_label}; plot skipped')
        return
    bp = ax.boxplot(datasets, positions=positions, widths=0.19,
                    patch_artist=True, showfliers=False, showmeans=True,
                    meanprops=dict(marker='D', markersize=4, markerfacecolor='white', markeredgecolor='black'),
                    medianprops=dict(color='black', linewidth=1.2))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    for patch, (_, j) in zip(bp['boxes'], labels):
        patch.set_facecolor(colors[j]); patch.set_alpha(0.65)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors[j], alpha=0.65, label=alg)
                       for j, alg in enumerate(ALGS)], ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_xticks([1, 2, 3], LEVELS)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Best MR violation severity (higher is better)')
    ax.set_xlabel('Perturbation magnitude (shared tertiles)')
    ax.grid(axis='y', alpha=0.2)
    ax.set_title(model_label)
    fig.tight_layout()
    target = outdir / f'{model_label.lower()}_rq2_shared_tertiles_boxplot.pdf'
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'D1/face_detection/sticker_detection')
    parser.add_argument('--run-start', type=int, default=2)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--expected-candidates', type=int, default=40)
    parser.add_argument('--strict-40', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('sticker_rq3_tertiles'),
                        help='Directory for the two PDF figures')
    args = parser.parse_args()
    root = args.root.expanduser()
    if not root.is_dir():
        parser.error(f'Root directory does not exist: {root}')
    for model, short_name in MODELS.items():
        print('\n' + '=' * 108)
        print(f'{short_name}: {model}')
        print('=' * 108)
        df, warnings = read_model(root, model, args.run_start, args.runs,
                                  args.expected_candidates, args.strict_40)
        units = df[['image_name', 'run']].drop_duplicates().shape[0]
        images = df.image_name.nunique()
        print(f'Paired image/run units: {units} | Unique images: {images}')
        print('Candidates per image/run/algorithm:')
        print(df.groupby(['algorithm', 'image_name', 'run']).size().groupby('algorithm')
              .agg(['min', 'max', 'mean']).to_string())
        q33, q67 = shared_tertiles(df)
        print(f'\nSHARED TERTILES: Q33 = {q33:.6f}, Q67 = {q67:.6f}')
        summary, image, total = analyze(df, q33, q67)
        print('\nMAIN TABLE — MATCHED IMAGE-LEVEL BEST MR SEVERITY')
        cols = ['level', 'algorithm', 'candidates', 'candidate_pct',
                'individual_coverage_pct', 'matched_units', 'matched_coverage_pct',
                'images', 'mean_best', 'median_best', 'sd_best', 'q25_best', 'q75_best']
        print(summary[cols].round(4).to_string(index=False))
        print('\nPAIRED WILCOXON TESTS (Holm correction within each magnitude category)')
        print(tests(image).to_string(index=False, float_format=lambda x: f'{x:.6g}'))
        save_boxplot(image, short_name, args.out.expanduser())
        if warnings:
            print(f'\nWARNINGS ({len(warnings)}):')
            for warning in warnings[:15]:
                print(' -', warning)
            if len(warnings) > 15:
                print(f' ... {len(warnings)-15} more warnings')
    print('\nOnly two PDF figures are saved; all numerical results are printed above.')
    print('Mean and SD are across IMAGE-level means of best MR score across matched runs.')


if __name__ == '__main__':
    main()
