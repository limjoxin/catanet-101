import numpy as np
import os
import pandas as pd
import argparse
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Adapted inspect script: visualize phase recognition bars (GT & Pred), remove experience plotting

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
phases_rev = phases[::-1]
n_phases = len(phases)

csv_files = glob.glob(os.path.join(args.input, '*.csv'))
all_df = []

for file in csv_files:
    vname = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    # compute ground-truth remaining surgical duration
    df['gt_rsd'] = df['elapsed'].max() - df['elapsed']
    df['video'] = vname
    all_df.append(df)

    # set up figure with 3 rows: RSD plot + GT bar + Pred bar, and legend on right
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 0.5, 0.5], width_ratios=[5, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])      # RSD plot
    ax_gt = fig.add_subplot(gs[1, 0])    # ground-truth phase bar
    ax_pred = fig.add_subplot(gs[2, 0])  # predicted phase bar
    ax_legend = fig.add_subplot(gs[:, 1])# phase-color legend

    # Plot RSD curves
    ax1.plot(df['elapsed'], df['gt_rsd'], label='Ground truth', linewidth=2)
    ax1.plot(df['elapsed'], df['predicted_rsd'], label='Predicted', linewidth=2)
    ax1.set_ylabel('RSD (min)')
    ax1.set_xlim(df['elapsed'].min(), df['elapsed'].max())
    ax1.set_ylim(0)
    ax1.legend()
    ax1.set_xticks([])

    # Determine bar height
    height = df['elapsed'].max() / 50

    # Ground truth phase bar
    ax_gt.imshow(
        df['ground_truth_step'].astype(int).values.reshape(1, -1),
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[0, df['elapsed'].max(), -height, height],
        vmin=0, vmax=n_phases-1
    )
    ax_gt.set_ylabel('GT')
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])

    # Predicted phase bar
    ax_pred.imshow(
        df['predicted_step'].astype(int).values.reshape(1, -1),
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[0, df['elapsed'].max(), -height, height],
        vmin=0, vmax=n_phases-1
    )
    ax_pred.set_ylabel('Pred')
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    # Legend for phases
    ax_legend.imshow(
        np.arange(n_phases)[:, None],
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[0, 1, 0, n_phases],
        vmin=0, vmax=n_phases-1
    )
    ax_legend.set_xticks([])
    ax_legend.set_yticks(np.arange(n_phases) + 0.5)
    ax_legend.set_yticklabels(phases_rev)
    ax_legend.yaxis.tick_right()

    # Save figure
    plt.savefig(os.path.join(args.input, f"{vname}.png"), bbox_inches='tight')
    plt.close()

# Concatenate all dataframes for metrics
all_df = pd.concat(all_df)
all_df['difference'] = all_df['gt_rsd'] - all_df['predicted_rsd']
all_df['absolute_error'] = np.abs(all_df['difference'])

def rsd_error(data_frame):
    rsd_2 = np.mean(data_frame[data_frame.gt_rsd < 2.0]['absolute_error'])
    rsd_5 = np.mean(data_frame[data_frame.gt_rsd < 5.0]['absolute_error'])
    rsd_all = np.mean(data_frame['absolute_error'])
    duration = np.max(data_frame['elapsed'])
    return pd.DataFrame([[rsd_2, rsd_5, rsd_all, duration]],
                         columns=['rsd_2', 'rsd_5', 'rsd_all', 'duration'])

rsd_err = all_df.groupby('video').apply(rsd_error)
rsd_err_s = pd.concat([rsd_err.mean(), rsd_err.std()], axis=1)
rsd_err_s.columns = ['mean', 'std']
print('\nMacro Average RSD')
print(rsd_err_s)

# Plot overall RSD error distribution
plt.figure()
plt.boxplot(rsd_err['rsd_all'])
plt.title('RSD Error')
plt.xticks([])
plt.ylabel('RSD MAE (min)')
plt.savefig(os.path.join(args.input, 'rsd_error_boxplot.png'), bbox_inches='tight')
plt.close()
