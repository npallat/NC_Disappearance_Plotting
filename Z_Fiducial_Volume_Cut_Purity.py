# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import polars as pl

# ********* USER INPUT ************
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
sample_POT = 5.30*(10**(20))
Run1_2_POT = 6.55*(10**(20))
POT = '6.55e20' # label
# *********************************

scale = Run1_2_POT/sample_POT

df = df.select(['signal.NC1p', 'rec.slc.reco.pfp.trk.len', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.npfp', 
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon', 'rec.slc.reco.pfp.trk.start.x',
			'rec.slc.reco.pfp.trk.start.y', 'rec.slc.reco.pfp.trk.start.z', 'rec.slc.reco.pfp.trk.end.x', 'rec.slc.reco.pfp.trk.end.y', 'rec.slc.reco.pfp.trk.end.z', 
			'rec.slc.barycenterFM.score', 'rec.mc.nu.prim.startE'])

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

# *********************************

def plot_with_errors(data, bins, label, color, scale=None, linestyle='solid'):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    if scale is not None:
        counts = counts * scale
        errors = errors * scale

    plt.stairs(counts, bin_edges, label=label, color=color, linestyle=linestyle)
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)

def plot_purity(signal, total, bins, label, color, linestyle='solid'):
    counts_pass, bin_edges = np.histogram(signal, bins=bins)
    counts_total, _ = np.histogram(total, bins=bins)

    pur = np.where(counts_total > 0, counts_pass / counts_total, 0)
    pur_err = np.where(
        counts_total > 0,
        np.sqrt(pur * (1 - pur) / counts_total),
        0
    )

    plt.stairs(pur, bin_edges, label=label, color=color, linestyle='solid')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.errorbar(bin_centers, pur, yerr=pur_err, fmt='none', ecolor=color, capsize=3)

# --------------------------------
colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())

# plot
fig_save = directory+'Z_Fiducial_Volume_varations_purity.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()

# z fiducial volume
'''
x_min = -190
x_max = 190
y_min = -190
y_max = 190
z_pairs = [[0, 500], [10, 500], [10, 490], [10, 480], [10, 470], [10, 460], [10, 450]]
i = 0
while i < len(z_pairs):
	z_min = z_pairs[i][0]
	z_max = z_pairs[i][1]
	signal_df = signal_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
	signal_df = signal_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
	plot_purity(signal_df['rec.slc.reco.pfp.trk.len'], df['rec.slc.reco.pfp.trk.len'], bins, str(z_min)+' < z < '+str(z_max), colors[i])
	i += 1
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#ax.text(0.58, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

'''

x_min = -190
x_max = 190
y_min = -190
y_max = 190
z_min = 10
z_max = 500
signal_df = signal_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
signal_df = signal_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))

cathode_crossing = df.filter(((pl.col('rec.slc.reco.pfp.trk.start.x') > 0) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 0)) | ((pl.col('rec.slc.reco.pfp.trk.start.x') < 0) & (pl.col('rec.slc.reco.pfp.trk.end.x') > 0)))
print(df.shape[0])
print(signal_df.shape[0])
print(cathode_crossing.shape[0])
print(cathode_crossing.filter(pl.col('signal.NC1p') == 1).shape[0])



