#!/usr/bin/env python3
"""RQ3 Face Detection: split half-violins for best MR severity and reported magnitude.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wilcoxon

ALGORITHMS = ('NSGA-II', 'SPEA2', 'MOEA/D')
DENSITIES = ('low_density', 'medium_density', 'high_density')
DIRS = {
    ('low_density', 'NSGA-II'): 'nsga_yunet_jaccard_2023_low/low_density_final',
    ('low_density', 'SPEA2'): 'spea2_yunet_jaccard_2023_low/low_density_final',
    ('low_density', 'MOEA/D'): 'moead_yunet_jaccard_2023_low/low_density_final',
    ('medium_density', 'NSGA-II'): 'nsga_yunet_jaccard_2023_medium/medium_density_final',
    ('medium_density', 'SPEA2'): 'spea2_yunet_jaccard_2023_medium/medium_density_final',
    ('medium_density', 'MOEA/D'): 'moead_yunet_jaccard_2023_medium/medium_density_final',
    ('high_density', 'NSGA-II'): 'nsga_yunet_jaccard_2023_high/high_density_final',
    ('high_density', 'SPEA2'): 'spea2_yunet_jaccard_2023_high/high_density_final',
    ('high_density', 'MOEA/D'): 'moead_yunet_jaccard_2023_high/high_density_final',
}
COLORS = {'severity': '#477F91', 'magnitude': '#D39A6A'}


def numeric(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (ValueError, TypeError):
        return np.nan


def load(root, first_run, n_runs):
    records, warnings = [], []
    for (density, algorithm), relative in DIRS.items():
        base = root / relative
        if not base.is_dir() and base.name.endswith('_final'):
            base = base.with_name(base.name[:-6])
        if not base.is_dir():
            raise FileNotFoundError(f'Experiment directory not found: {base}')
        for run in range(first_run, first_run + n_runs):
            folder = base / f'run_{run:02d}'
            paths = sorted(folder.glob('*/summary.json'))
            if not paths:
                warnings.append(f'No summaries: {folder}')
            for path in paths:
                try:
                    s = json.loads(path.read_text(encoding='utf-8'))
                    name = s.get('image_name')
                    if not name:
                        raise ValueError('missing image_name')
                    method = str(s.get('method', '')).lower()
                    tokens = {'NSGA-II': ('nsga',), 'SPEA2': ('spea',), 'MOEA/D': ('moead', 'moea/d')}
                    if method and not any(t in method for t in tokens[algorithm]):
                        raise ValueError(f'method mismatch: {method}')
                    severity = numeric(s.get('best_saved_mr_score'))
                    magnitude = numeric(s.get('best_saved_magnitude'))
                    if not np.isfinite(severity) or not np.isfinite(magnitude):
                        raise ValueError('missing or nonnumeric severity/magnitude')
                    if not (0 <= severity <= 1 and 0 <= magnitude <= 1):
                        raise ValueError(f'out of expected [0,1] range: {severity}, {magnitude}')
                    records.append(dict(density=density, image_name=str(name), algorithm=algorithm,
                                        run=run, severity=severity, magnitude=magnitude,
                                        summary_path=str(path)))
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    warnings.append(f'Skipped {path}: {e}')
    if not records:
        raise RuntimeError('No usable summary.json records found')
    df = pd.DataFrame(records)
    keys = ['density', 'image_name', 'algorithm', 'run']
    if df.duplicated(keys).any():
        raise RuntimeError('Duplicate image/algorithm/run summaries found')
    return df, warnings


def select_complete(df, n_runs):
    counts = df.groupby(['density', 'image_name', 'algorithm']).run.nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=ALGORITHMS, fill_value=0)
    valid = counts.eq(n_runs).all(axis=1)
    if not valid.any():
        raise RuntimeError('No images have complete runs for all three MOEAs')
    print(f'Fully paired images: {valid.sum()} of {len(valid)}')
    return df.set_index(['density', 'image_name']).loc[counts.index[valid]].reset_index()


def draw_half(ax, values, x, side, color, width=0.37, bw=0.22):
    """Half violin with an inset miniature boxplot (Q1, median, Q3, 1.5-IQR whiskers)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    y = np.linspace(0, 1, 501)
    if len(values) >= 3 and np.std(values) > 1e-7:
        kde = gaussian_kde(values, bw_method=bw)
        density = kde(y)
        density /= max(float(density.max()), 1e-12)
        ax.fill_betweenx(y, x, x + side * width * density,
                         facecolor=color, alpha=0.78, edgecolor=color,
                         linewidth=0.9, zorder=2)
    else:
        ax.scatter(np.full(len(values), x + side * width * .35), values,
                   s=12, color=color, zorder=3)

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    whisker_low = float(inside.min()) if len(inside) else float(q1)
    whisker_high = float(inside.max()) if len(inside) else float(q3)

    # Position the miniature box close to the central split line.
    center = x + side * 0.065
    box_width = 0.046
    ink = '#23323B'
    ax.plot([center, center], [whisker_low, whisker_high], color=ink,
            linewidth=0.85, zorder=5)
    cap = box_width * 0.65
    for endpoint in (whisker_low, whisker_high):
        ax.plot([center - cap/2, center + cap/2], [endpoint, endpoint],
                color=ink, linewidth=0.85, zorder=5)
    ax.add_patch(Rectangle((center - box_width/2, q1), box_width, max(q3-q1, 0.002),
                           facecolor='white', edgecolor=ink, linewidth=1.05, zorder=6))
    ax.plot([center - box_width/2, center + box_width/2], [median, median],
            color=ink, linewidth=1.45, zorder=7)
    # Mean: small diamond, drawn on top of the miniature boxplot.
    ax.scatter([center], [float(np.mean(values))], marker='D', s=19,
               facecolor='#B34F47', edgecolor='white', linewidth=0.65, zorder=9)


def plot(image, out, bw, dpi):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 10.5, 'ytick.labelsize': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, algorithm in enumerate(ALGORITHMS):
        sub = image[image.algorithm == algorithm]
        draw_half(ax, sub.severity.to_numpy(), i, -1, COLORS['severity'], bw=bw)
        draw_half(ax, sub.magnitude.to_numpy(), i, +1, COLORS['magnitude'], bw=bw)
        ax.plot([i, i], [0, 1], color='#38434B', linewidth=0.55, alpha=0.55, zorder=4)
    ax.set_xlim(-0.53, 2.53)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel('Score', labelpad=9)
    ax.set_xticks(range(3), ALGORITHMS)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=11, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.grid(axis='y', color='#E9ECEF', linewidth=0.65)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=COLORS['severity'], alpha=.78, label='Violation severity (MR score) ↑'),
        Patch(facecolor=COLORS['magnitude'], alpha=.78, label='Perturbation magnitude ↓')],
        loc='upper center', bbox_to_anchor=(0.5, -0.095), ncol=2,
        frameon=False, handlelength=1.35, columnspacing=2.5, fontsize=9.5)
    fig.subplots_adjust(left=.09, right=.99, top=.87, bottom=.21)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'yunet_rq2_split_violin.{ext}', dpi=dpi,
                    bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def holm(pvalues):
    order = np.argsort(pvalues)
    n = len(pvalues)
    adjusted = np.empty(n, dtype=float)
    prev = 0.0
    for rank, ix in enumerate(order):
        prev = max(prev, (n - rank) * pvalues[ix])
        adjusted[ix] = min(1.0, prev)
    return adjusted


def paired_tests(image):
    rows = []
    pairs = [('NSGA-II', 'SPEA2'), ('NSGA-II', 'MOEA/D'), ('SPEA2', 'MOEA/D')]
    for metric in ('severity', 'magnitude'):
        wide = image.pivot(index=['density', 'image_name'], columns='algorithm', values=metric)
        batch = []
        for a, b in pairs:
            pair = wide[[a, b]].dropna()
            d = (pair[a] - pair[b]).to_numpy()
            nonzero = d[~np.isclose(d, 0, atol=1e-12)]
            p = float(wilcoxon(nonzero, alternative='two-sided').pvalue) if len(nonzero) else 1.0
            batch.append(dict(metric=metric, algorithm_a=a, algorithm_b=b,
                              n_images=len(pair), mean_paired_difference=float(np.mean(d)),
                              median_paired_difference=float(np.median(d)), p_raw=p))
        adjusted = holm(np.array([r['p_raw'] for r in batch]))
        for row, p in zip(batch, adjusted):
            row['p_holm'] = p
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.home() / 'D1/face_detection/WIDER_val_density_600')
    p.add_argument('--out', type=Path, default=Path('face_rq3_violin_results'))
    p.add_argument('--run-start', type=int, default=1)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--bandwidth', type=float, default=.22,
                   help='KDE bandwidth multiplier (default 0.22); same for both metrics and all algorithms')
    p.add_argument('--dpi', type=int, default=400)
    args = p.parse_args()
    if args.runs < 1 or args.bandwidth <= 0:
        p.error('--runs and --bandwidth must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    raw, warnings = load(args.root.expanduser(), args.run_start, args.runs)
    raw.to_csv(args.out / 'all_valid_run_summaries.csv', index=False)
    selected = select_complete(raw, args.runs)
    selected.to_csv(args.out / 'selected_runs.csv', index=False)
    image = selected.groupby(['density', 'image_name', 'algorithm'], as_index=False)[['severity', 'magnitude']].mean()
    image.to_csv(args.out / 'image_level_metrics.csv', index=False)
    summary = image.groupby('algorithm')[['severity', 'magnitude']].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    summary = summary.reindex(ALGORITHMS)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.out / 'descriptive_statistics.csv', index=False)

    # Print the exact values represented by the violins and miniature boxplots.
    # Each observation is one image-level mean over the selected independent runs.
    lines = [
        '',
        'RQ2 — YuNet: image-level descriptive statistics',
        f'Complete paired images: {image[["density", "image_name"]].drop_duplicates().shape[0]}',
        'Each image contributes one mean across the selected runs per algorithm.',
        '',
        f'{"Algorithm":<10} {"Metric":<25} {"N":>5} {"Mean":>10} {"Median":>10} {"SD":>10} {"Q1":>10} {"Q3":>10} {"Min":>10} {"Max":>10}',
        '-' * 112,
    ]
    detailed_rows = []
    for algorithm in ALGORITHMS:
        sub = image.loc[image['algorithm'] == algorithm]
        for metric, label in [('severity', 'MR violation severity'),
                              ('magnitude', 'Perturbation magnitude')]:
            vals = sub[metric].dropna().to_numpy(dtype=float)
            q1, q3 = np.percentile(vals, [25, 75])
            stats = dict(algorithm=algorithm, metric=metric, n=len(vals),
                         mean=float(np.mean(vals)), median=float(np.median(vals)),
                         sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                         q1=float(q1), q3=float(q3),
                         minimum=float(np.min(vals)), maximum=float(np.max(vals)))
            detailed_rows.append(stats)
            lines.append(
                f'{algorithm:<10} {label:<25} {stats["n"]:>5d} '
                f'{stats["mean"]:>10.6f} {stats["median"]:>10.6f} '
                f'{stats["sd"]:>10.6f} {stats["q1"]:>10.6f} '
                f'{stats["q3"]:>10.6f} {stats["minimum"]:>10.6f} '
                f'{stats["maximum"]:>10.6f}'
            )
        lines.append('')
    report = '\n'.join(lines)
    print(report, flush=True)
    (args.out / 'descriptive_statistics_report.txt').write_text(report + '\n', encoding='utf-8')
    pd.DataFrame(detailed_rows).to_csv(args.out / 'descriptive_statistics_detailed.csv', index=False)

    paired_tests(image).to_csv(args.out / 'pairwise_moea_wilcoxon_holm.csv', index=False)
    (args.out / 'validation_warnings.txt').write_text('\n'.join(warnings) + ('\n' if warnings else ''))
    plot(image, args.out, args.bandwidth, args.dpi)
    print(f'Outputs: {args.out.resolve()}')
    print('Interpretation: halves are marginal distributions, NOT paired solution-level score/magnitude observations.')
    print('Check the implementation definition of best_saved_magnitude before describing it as magnitude of the highest-MR solution.')


if __name__ == '__main__':
    main()
