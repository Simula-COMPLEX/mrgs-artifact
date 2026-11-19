#!/usr/bin/env python3
"""Screw detection RQ2 failure profiles (origSCDM as SCDM1, noscrewSCDM as SCDM2, fixtureSCDM as SCDM3).

Three-panel figures per model and combined (equal model weight):
MR1 detection counts (matched/lost/new) + thin zero-match candidate bar;
MR2/MR3 candidate-level severity conditional on >=1 matched detection.

Input: <experiment>/run_XX/<image>/ALL_SAVED/saved_candidates.csv
Runs: run_01 through run_10 by default.
Output: PDF and SVG per model and combined, plus figure_summary.csv.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ALGORITHMS = ('NSGA-II', 'SPEA2', 'MOEA/D')
MODELS = ('origSCDM', 'noscrewSCDM', 'fixtureSCDM')
ROOT = Path.home() / 'D1' / 'face_detection' / 'screw_detection'
DIRECTORIES = {
    'origSCDM': {
        'NSGA-II': 'initial_dataset/nsga2_jaccard_screw_orig/all_train',
        'SPEA2': 'initial_dataset/spea2_jaccard_screw_orig/all_train',
        'MOEA/D': 'initial_dataset/moead_jaccard_screw_orig/all_train',
    },
    'noscrewSCDM': {
        'NSGA-II': 'added_more_noscrews_at_diff_exposure/nsga2_jaccard_noscrews/all_train',
        'SPEA2': 'added_more_noscrews_at_diff_exposure/spea2_jaccard_noscrews/all_train',
        'MOEA/D': 'added_more_noscrews_at_diff_exposure/moead_jaccard_noscrews/all_train',
    },
    'fixtureSCDM': {
        'NSGA-II': 'added_screw_fixture/nsga2_jaccard_fixture/all_train',
        'SPEA2': 'added_screw_fixture/spea2_jaccard_fixture/all_train',
        'MOEA/D': 'added_screw_fixture/moead_jaccard_fixture/all_train',
    },
}
RESULT_DIRECTORIES = {(model, algorithm): DIRECTORIES[model][algorithm]
                      for model in MODELS for algorithm in ALGORITHMS}

COUNTS = ('matched_after', 'lost_predictions', 'new_predictions')
DEVIATIONS = ('confidence_violation', 'localization_violation')
COLORS = ('#6D99A8', '#DEA675', '#C97371')
FAIL_COLOR = '#85639A'


def resolve_dirs(root):
    dirs = {(model, algorithm): root / relative
            for (model, algorithm), relative in RESULT_DIRECTORIES.items()}
    for (model, algorithm), base in dirs.items():
        if not base.is_dir():
            raise FileNotFoundError(f'Missing {model} / {algorithm} experiment directory: {base}')
        print(f'[FOUND] {model} / {algorithm}: {base}', flush=True)
    return dirs


def load_data(root, first_run, runs):
    dirs = resolve_dirs(root)
    frames, available, discovered, issues = [], set(), set(), []
    for (model, algorithm), base in dirs.items():
        for run in range(first_run, first_run + runs):
            run_dir = base / f'run_{run:02d}'
            if not run_dir.is_dir():
                raise FileNotFoundError(f'Missing run directory: {run_dir}')
            for image_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                image = image_dir.name
                discovered.add((model, image))
                path = image_dir / 'ALL_SAVED' / 'saved_candidates.csv'
                if not path.is_file():
                    continue
                try:
                    df = pd.read_csv(path)
                    missing = set(COUNTS + DEVIATIONS) - set(df.columns)
                    if missing:
                        raise ValueError(f'missing columns: {sorted(missing)}')
                    if df.empty:
                        raise ValueError('empty candidate CSV')
                    for col in COUNTS + DEVIATIONS:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    counts = df[list(COUNTS)].to_numpy(dtype=float)
                    valid = np.isfinite(counts).all(axis=1)
                    valid &= (counts >= 0).all(axis=1)
                    valid &= np.isclose(counts, np.round(counts)).all(axis=1)
                    if (~valid).any():
                        issues.append(f'{path}: excluded {int((~valid).sum())} rows with invalid detection counts')
                    df = df.loc[valid, list(COUNTS + DEVIATIONS)].copy()
                    if df.empty:
                        raise ValueError('no rows with valid detection counts')
                    for col in COUNTS:
                        df[col] = df[col].astype(int)
                    for col in DEVIATIONS:
                        df.loc[~np.isfinite(df[col]) | (df[col] < 0), col] = np.nan
                    df['model'], df['algorithm'], df['image_name'], df['run'] = model, algorithm, image, run
                    if (model, image, algorithm, run) in available:
                        raise ValueError('duplicate image/run candidate CSV')
                    frames.append(df)
                    available.add((model, image, algorithm, run))
                except (OSError, ValueError, pd.errors.ParserError) as exc:
                    issues.append(f'{path}: {exc}')
    if not frames:
        raise RuntimeError('No usable saved_candidates.csv files found')
    return pd.concat(frames, ignore_index=True), available, discovered, issues


def pair_images(df, available, discovered, first_run, runs):
    expected = range(first_run, first_run + runs)
    paired = {(model, image) for model, image in discovered
              if all((model, image, alg, run) in available
                     for alg in ALGORITHMS for run in expected)}
    print('\nPaired image coverage (all algorithms x all runs, separately by model):')
    for model in MODELS:
        count = sum(m == model for m, _ in paired)
        total = sum(m == model for m, _ in discovered)
        print(f'  {model}: {count}/{total}')
    if not paired:
        raise RuntimeError('No paired images found')
    keys = pd.MultiIndex.from_tuples(sorted(paired), names=['model', 'image_name'])
    mask = pd.MultiIndex.from_frame(df[['model','image_name']]).isin(keys)
    return df.loc[mask].copy()


def image_run_percentages(df):
    rows = []
    for (model, image, algorithm, run), group in df.groupby(
            ['model','image_name','algorithm','run'], sort=False):
        rec = dict(model=model, image_name=image, algorithm=algorithm, run=run,
                   n_candidates=len(group))
        # MR1 counts all valid saved candidates, including zero-match candidates.
        outcome = group[list(COUNTS)].sum()
        total = int(outcome.sum())
        for col, short in zip(COUNTS, ('matched','lost','new')):
            rec[f'mr1_{short}'] = 100 * outcome[col] / total if total else np.nan
        # The purple bar is a CANDIDATE-level percentage. Both-empty candidates
        # are excluded from the denominator (not classified as complete failure).
        union = group[list(COUNTS)].sum(axis=1)
        nonempty = union > 0
        zero_match = nonempty & (group['matched_after'] == 0)
        rec['complete_failure'] = (100 * zero_match.sum() / nonempty.sum()
                                   if nonempty.any() else np.nan)
        rec['n_nonempty_candidates'] = int(nonempty.sum())
        rec['n_zero_match_candidates'] = int(zero_match.sum())
        for mr, col in [('mr2','confidence_violation'),('mr3','localization_violation')]:
            eligible = (group['matched_after'] > 0) & group[col].notna()
            vals = group.loc[eligible, col]
            rec[f'{mr}_n_evaluable'] = len(vals)
            rec[f'{mr}_small'] = 100 * (vals <= .05).mean() if len(vals) else np.nan
            rec[f'{mr}_moderate'] = 100 * ((vals > .05) & (vals <= .10)).mean() if len(vals) else np.nan
            rec[f'{mr}_large'] = 100 * (vals > .10).mean() if len(vals) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(per_run, model):
    sub = per_run if model == 'all_models' else per_run[per_run.model == model]
    # Run-level rates -> image-level means -> model means (combined: models equal).
    cols = ['mr1_matched','mr1_lost','mr1_new','complete_failure',
            'mr2_small','mr2_moderate','mr2_large',
            'mr3_small','mr3_moderate','mr3_large']
    image = sub.groupby(['model','image_name','algorithm'], as_index=False)[cols].mean()
    if model == 'all_models':
        model_mean = image.groupby(['model','algorithm'], as_index=False)[cols].mean()
        result = model_mean.groupby('algorithm', as_index=False)[cols].mean()
    else:
        result = image.groupby('algorithm', as_index=False)[cols].mean()
    result['model'] = model
    result['n_images'] = result.algorithm.map(image.groupby('algorithm').size())
    # These are informative coverage counts, not denominators for plotted macro means.
    coverage = sub.groupby('algorithm', as_index=False).agg(
        n_candidates=('n_candidates','sum'),
        n_zero_match_candidates=('n_zero_match_candidates','sum'),
        n_mr2_evaluable=('mr2_n_evaluable','sum'),
        n_mr3_evaluable=('mr3_n_evaluable','sum'))
    return result.merge(coverage, on='algorithm', how='left')


def displayed_percentages(values):
    """Round three nonnegative proportions to 0.01% with an exact 100.00% sum.

    Largest-remainder allocation changes only text labels, not plotted values.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        return None
    exact = values / values.sum() * 10000  # hundredths of one percent
    units = np.floor(exact).astype(int)
    remainder = 10000 - int(units.sum())
    for j in np.argsort(-(exact - units), kind='stable')[:remainder]:
        units[j] += 1
    return units / 100.0


def draw_panel(ax, summary, keys, title, legend, colors, show_failure=False):
    y = np.arange(len(ALGORITHMS)) * 1.23
    bar_height = .38 if show_failure else .48
    for i, alg in enumerate(ALGORITHMS):
        row = summary.loc[summary.algorithm == alg]
        if row.empty:
            continue
        values = row.iloc[0][keys].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            ax.text(50,y[i], 'Not evaluable', va='center',ha='center',fontsize=8)
        else:
            labels = displayed_percentages(values)
            left = 0.0
            for v,label,color in zip(values,labels,colors):
                ax.barh(y[i],v,left=left,height=bar_height,color=color,
                        edgecolor='white',linewidth=.6)
                if v >= 12:
                    ax.text(left+v/2,y[i],f'{label:.1f}',ha='center',va='center',
                            fontsize=8,fontweight='bold',color='white')
                left += v
        if show_failure:
            fail = float(row.iloc[0]['complete_failure'])
            if np.isfinite(fail):
                ax.barh(y[i]+.36,100,height=.105,color='#EAE8EC',edgecolor='none')
                ax.barh(y[i]+.36,fail,height=.105,color=FAIL_COLOR,edgecolor='none')
                ax.text(min(fail+2.0,91),y[i]+.36,f'{fail:.1f}',
                        fontsize=7,color=FAIL_COLOR,va='center',fontweight='bold')
    ax.set_yticks(y,ALGORITHMS)
    ax.invert_yaxis()
    ax.set_ylim(y[-1]+.68,-.48)
    ax.set_xlim(0,100)
    ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%'])
    ax.set_title(title,fontsize=11,fontweight='bold',pad=13)
    ax.tick_params(axis='both',length=0,labelsize=9)
    ax.grid(axis='x',alpha=.15)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    patches = [Patch(facecolor=color,label=label) for color,label in zip(colors,legend)]
    if show_failure:
        patches.append(Patch(facecolor=FAIL_COLOR,label='Zero-match'))
    ax.legend(handles=patches,loc='upper center',bbox_to_anchor=(.5,-.14),
              frameon=False,ncol=4 if show_failure else 3,fontsize=7.1,
              columnspacing=.9,handlelength=1.2)


def plot_figure(summary, model, out, dpi, png):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,3,figsize=(13.7,3.45),sharey=True)
    draw_panel(axes[0],summary,['mr1_matched','mr1_lost','mr1_new'],
               'MR1 · Detection preservation',('Matched','Lost','New'),COLORS,True)
    draw_panel(axes[1],summary,['mr2_small','mr2_moderate','mr2_large'],
               'MR2 · Confidence stability',('Small','Moderate','Large'),COLORS)
    draw_panel(axes[2],summary,['mr3_small','mr3_moderate','mr3_large'],
               'MR3 · Localization stability',('Small','Moderate','Large'),COLORS)
    fig.subplots_adjust(left=.085,right=.985,top=.85,bottom=.29,wspace=.26)
    for ext in (('pdf','svg','png') if png else ('pdf','svg')):
        fig.savefig(out/f'screw_rq2_failure_profiles_{model}.{ext}',
                    dpi=dpi,bbox_inches='tight',pad_inches=.14)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT, help='Screw detection experiment root')
    p.add_argument('--out', type=Path, default=Path('screw_rq2_failure_profiles'))
    p.add_argument('--run-start', type=int, default=1, help='First run to include (default: 1)')
    p.add_argument('--runs', type=int, default=10, help='Number of runs (default: 10, selecting run_01 through run_10)')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--png', action='store_true')
    args = p.parse_args()
    if args.runs < 1:
        p.error('--runs must be positive')
    root = args.root.expanduser().resolve()
    out = args.out.expanduser()
    df, available, discovered, issues = load_data(root, args.run_start, args.runs)
    selected = pair_images(df, available, discovered, args.run_start, args.runs)
    per_run = image_run_percentages(selected)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in (*MODELS, 'all_models'):
        summary = summarize(per_run, model)
        if summary.empty:
            continue
        summaries.append(summary)
        plot_figure(summary, model, out, args.dpi, args.png)
        print(f'\n{model}:')
        print(summary[['algorithm','n_images','complete_failure','mr1_lost',
                       'mr1_new','mr2_large','mr3_large']].round(2).to_string(index=False))
    pd.concat(summaries, ignore_index=True).to_csv(out/'figure_summary.csv', index=False)
    print(f'\nSkipped/invalid input notes: {len(issues)} (not saved)')
    for issue in issues[:8]:
        print(' ', issue)
    if len(issues) > 8:
        print(f'  ... and {len(issues)-8} more')
    print(f'\nFigures and one summary CSV saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
