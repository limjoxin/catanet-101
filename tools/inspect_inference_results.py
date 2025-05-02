import numpy as np
import os
import pandas as pd
import argparse
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

# Script: visualize RSD & phases, compute precision/recall/F1 and per-phase accuracy per-case and overall,
# generate confusion matrices and classification reports per-case and overall.

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', type=str, required=True,
    help='Path to folder containing per-video inference CSVs'
)
args = parser.parse_args()

plt.rcParams['font.size'] = 16
cmap = plt.colormaps['tab20']  # discrete colors for phases

# Define phase mapping
PHASE_MAPPING = {
    'Incision': 0,
    'Viscoelastic': 1,
    'Capsulorhexis': 2,
    'Hydrodissection': 3,
    'Phacoemulsification': 4,
    'Irrigation/Aspiration': 5,
    'Capsule Pulishing': 6,
    'Lens Implantation': 7,
    'Lens positioning': 8,
    'Anterior-Chamber Flushing': 9,
    'Viscoelastic_Suction': 10,
    'Tonifying/Antibiotics': 11,
    'Idle': 12
}
# Derive ordered list of phases and labels
phases = [phase for phase, idx in sorted(PHASE_MAPPING.items(), key=lambda x: x[1])]
n_phases = len(phases)
labels = list(range(n_phases))

# Gather CSV files, filter out inference CSVs only (skip metrics, reports, and matrices)
all_csvs = glob.glob(os.path.join(args.input, '*.csv'))
csv_files = []
for f in all_csvs:
    base = os.path.basename(f)
    # skip generated summaries and reports
    if base.startswith(('metrics_per_video', 'metrics_overall')):
        continue
    if 'confusion_matrix' in base or 'classification_report' in base:
        continue
    csv_files.append(f)

all_df_list = []
per_video_metrics = []

for file_path in csv_files:
    vname = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)

    # Compute GT RSD and tag video
    df['gt_rsd'] = df['elapsed'].max() - df['elapsed']
    df['video'] = vname

    # Visualization: RSD curves + phase bars
    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(3,2, height_ratios=[2.5,0.5,0.5], width_ratios=[5,1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0,0]); ax_gt  = fig.add_subplot(gs[1,0])
    ax_pr = fig.add_subplot(gs[2,0]); ax_leg = fig.add_subplot(gs[:,1])
    ax1.plot(df['elapsed'], df['gt_rsd'],        label='GT RSD', linewidth=2)
    ax1.plot(df['elapsed'], df['predicted_rsd'], label='Pred RSD', linewidth=2)
    ax1.set_ylabel('RSD (min)'); ax1.legend(); ax1.set_xticks([])
    bar_h = df['elapsed'].max()/50.0
    for ax,col,lbl in [(ax_gt,'ground_truth_step','GT'),(ax_pr,'predicted_step','Pred')]:
        ax.imshow(df[col].astype(int).values.reshape(1,-1), aspect='auto', cmap=cmap,
                  interpolation='nearest', extent=[0,df['elapsed'].max(),-bar_h,bar_h],
                  vmin=0, vmax=n_phases-1)
        ax.set_ylabel(lbl); ax.set_xticks([]); ax.set_yticks([])
    ax_leg.imshow(np.arange(n_phases)[:,None], aspect='auto', cmap=cmap,
                  interpolation='nearest', extent=[0,1,0,n_phases], vmin=0,vmax=n_phases-1)
    ax_leg.set_xticks([]); ax_leg.set_yticks(np.arange(n_phases)+0.5)
    ax_leg.set_yticklabels(phases[::-1]); ax_leg.yaxis.tick_right()
    fig.savefig(os.path.join(args.input, f"{vname}.png"), bbox_inches='tight')
    plt.close(fig)

    # Per-case classification metrics
    y_true = df['ground_truth_step']; y_pred = df['predicted_step']
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Plot confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8,6))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(labels); ax_cm.set_xticklabels(phases, rotation=45, ha='right')
    ax_cm.set_yticks(labels); ax_cm.set_yticklabels(phases)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Ground Truth')
    ax_cm.set_title(f'Confusion Matrix: {vname}')
    fig_cm.savefig(os.path.join(args.input, f"{vname}_confusion_matrix.png"), bbox_inches='tight')
    plt.close(fig_cm)

    # Classification report text and CSV
    report_txt = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=phases,
        zero_division=0
    )
    # save text
    with open(os.path.join(args.input, f"{vname}_classification_report.txt"), 'w') as f:
        f.write(report_txt)
    # save CSV
    rep_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=phases,
        output_dict=True,
        zero_division=0
    )
    rep_df = pd.DataFrame(rep_dict).T
    rep_df.to_csv(os.path.join(args.input, f"{vname}_classification_report.csv"))

    # Collect for aggregate metrics
    per_video_metrics.append(rep_df.assign(video=vname))
    all_df_list.append(df)

# Aggregate all data
all_df = pd.concat(all_df_list, ignore_index=True)

# Overall classification metrics
y_true_all = all_df['ground_truth_step']; y_pred_all = all_df['predicted_step']
# Confusion matrix
cm_all = confusion_matrix(y_true_all, y_pred_all, labels=labels)
fig_all, ax_all = plt.subplots(figsize=(8,6))
im_all = ax_all.imshow(cm_all, interpolation='nearest', cmap='Blues')
plt.colorbar(im_all, ax=ax_all)
ax_all.set_xticks(labels); ax_all.set_xticklabels(phases, rotation=45, ha='right')
ax_all.set_yticks(labels); ax_all.set_yticklabels(phases)
ax_all.set_xlabel('Predicted'); ax_all.set_ylabel('Ground Truth')
ax_all.set_title('Confusion Matrix: All Cases')
fig_all.savefig(os.path.join(args.input, 'confusion_matrix_all.png'), bbox_inches='tight')
plt.close(fig_all)

# Overall classification report
report_txt_all = classification_report(
    y_true_all, y_pred_all,
    labels=labels,
    target_names=phases,
    zero_division=0
)
with open(os.path.join(args.input, 'classification_report_all.txt'), 'w') as f:
    f.write(report_txt_all)
rep_dict_all = classification_report(
    y_true_all, y_pred_all,
    labels=labels,
    target_names=phases,
    output_dict=True,
    zero_division=0
)
rep_df_all = pd.DataFrame(rep_dict_all).T
rep_df_all.to_csv(os.path.join(args.input, 'classification_report_all.csv'))

print("Saved per-case and overall classification reports and confusion matrices.")




