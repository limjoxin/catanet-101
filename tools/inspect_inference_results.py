import numpy as np
import os
import pandas as pd
import argparse
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Updated visualization: RSD + GT & Pred phase bars, removed experience plotting,
# plus confusion matrix and classification metrics per phase, per video, and overall.

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type=str,
    help='path to folder containing test results in csv'
)
args = parser.parse_args()

plt.rcParams['font.size'] = 16
cmap = plt.colormaps['tab20']  # discrete colors for phases

# Define phase names
phases = ['None', 'Inc', 'VisAgInj', 'Rhexis', 'Hydro', 'Phaco',
          'IrAsp', 'CapsPol', 'LensImpl', 'VisAgRem', 'TonAnti']
n_phases = len(phases)
labels = list(range(n_phases))

csv_files = glob.glob(os.path.join(args.input, '*.csv'))
all_df = []

# Process each video
for file in csv_files:
    vname = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    # Compute ground-truth remaining surgical duration
    df['gt_rsd'] = df['elapsed'].max() - df['elapsed']
    if 'ground_truth_step' not in df.columns:
        raise KeyError("CSV file does not contain 'ground_truth_step' column")
    df['video'] = vname
    all_df.append(df)

    # Plot RSD and phase bars
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 0.5, 0.5], width_ratios=[5, 1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[2, 0])
    ax_leg = fig.add_subplot(gs[:, 1])

    ax1.plot(df['elapsed'], df['gt_rsd'], label='GT RSD', linewidth=2)
    ax1.plot(df['elapsed'], df['predicted_rsd'], label='Pred RSD', linewidth=2)
    ax1.set_ylabel('RSD (min)')
    ax1.set_xlim(df['elapsed'].min(), df['elapsed'].max())
    ax1.set_ylim(0)
    ax1.legend()
    ax1.set_xticks([])

    height = df['elapsed'].max() / 50
    ax_gt.imshow(df['ground_truth_step'].astype(int).values.reshape(1, -1),
                 aspect='auto', cmap=cmap, interpolation='nearest',
                 extent=[0, df['elapsed'].max(), -height, height], vmin=0, vmax=n_phases-1)
    ax_gt.set_ylabel('GT')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    ax_pred.imshow(df['predicted_step'].astype(int).values.reshape(1, -1),
                   aspect='auto', cmap=cmap, interpolation='nearest',
                   extent=[0, df['elapsed'].max(), -height, height], vmin=0, vmax=n_phases-1)
    ax_pred.set_ylabel('Pred')
    ax_pred.set_xticks([]); ax_pred.set_yticks([])

    ax_leg.imshow(np.arange(n_phases)[:, None], aspect='auto', cmap=cmap,
                  interpolation='nearest', extent=[0, 1, 0, n_phases],
                  vmin=0, vmax=n_phases-1)
    ax_leg.set_xticks([])
    ax_leg.set_yticks(np.arange(n_phases) + 0.5)
    ax_leg.set_yticklabels(phases[::-1])
    ax_leg.yaxis.tick_right()

    plt.savefig(os.path.join(args.input, f"{vname}.png"), bbox_inches='tight')
    plt.close()

# Aggregate all videos
all_df = pd.concat(all_df)

# Compute RSD error metrics
all_df['abs_error'] = np.abs(all_df['gt_rsd'] - all_df['predicted_rsd'])
def rsd_error(df):
    return pd.Series({
        'rsd_2_mean': df[df.gt_rsd<2].abs_error.mean(),
        'rsd_5_mean': df[df.gt_rsd<5].abs_error.mean(),
        'rsd_all_mean': df.abs_error.mean(),
        'duration': df.elapsed.max()
    })
rsd_err = all_df.groupby('video').apply(rsd_error)
rsd_summary = pd.concat([rsd_err.mean(), rsd_err.std()], axis=1)
rsd_summary.columns = ['mean', 'std']
print("\nMacro Average RSD:\n", rsd_summary)

plt.figure()
plt.boxplot(rsd_err['rsd_all_mean'])
plt.title('RSD Error')
plt.ylabel('MAE (min)')
plt.xticks([])
plt.savefig(os.path.join(args.input, 'rsd_error_boxplot.png'), bbox_inches='tight')
plt.close()

# Confusion matrix across all frames
y_true_all = all_df['ground_truth_step']
y_pred_all = all_df['predicted_step']
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(labels); ax.set_xticklabels(phases, rotation=45, ha='right')
ax.set_yticks(labels); ax.set_yticklabels(phases)
ax.set_xlabel('Predicted phase'); ax.set_ylabel('Ground truth phase')
ax.set_title('Confusion Matrix (All Videos)')
plt.tight_layout()
plt.savefig(os.path.join(args.input, 'phase_confusion_matrix.png'), bbox_inches='tight')
plt.close()

# Compute classification metrics overall
acc_overall = accuracy_score(y_true_all, y_pred_all)
prec_overall, rec_overall, f1_overall, sup_overall = \
    precision_recall_fscore_support(y_true_all, y_pred_all, labels=labels, zero_division=0)
metrics_overall = pd.DataFrame({
    'phase': phases,
    'precision': prec_overall,
    'recall': rec_overall,
    'f1_score': f1_overall,
    'support': sup_overall
})
metrics_overall.loc[len(metrics_overall)] = ['accuracy', acc_overall, acc_overall, acc_overall, len(y_true_all)]
# Add macro average row
metrics_overall.loc[len(metrics_overall)] = [
    'macro_avg',
    np.mean(prec_overall),
    np.mean(rec_overall),
    np.mean(f1_overall),
    ''
]
metrics_overall.to_csv(os.path.join(args.input, 'metrics_overall.csv'), index=False)
print("Saved overall metrics to metrics_overall.csv")

# Compute classification metrics per video
per_video = []
for vid, grp in all_df.groupby('video'):
    yt = grp['ground_truth_step']; yp = grp['predicted_step']
    acc = accuracy_score(yt, yp)
    p, r, f1, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    for i, phase in enumerate(phases):
        per_video.append({
            'video': vid,
            'phase': phase,
            'precision': p[i],
            'recall': r[i],
            'f1_score': f1[i],
            'support': s[i]
        })
    per_video.append({
        'video': vid,
        'phase': 'accuracy',
        'precision': acc,
        'recall': acc,
        'f1_score': acc,
        'support': len(yt)
    })
per_video_df = pd.DataFrame(per_video)
per_video_df.to_csv(os.path.join(args.input, 'metrics_per_video.csv'), index=False)
print("Saved per-video metrics to metrics_per_video.csv")

