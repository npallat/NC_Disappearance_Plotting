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
NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/cosmic_bg/'
# *********************************

NC1p_df = NC1p_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in NC1p_df.columns])
NC1p_df = NC1p_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 500))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 500))

# *********************************

def plot_with_errors(data, bins, label, color):
	counts, bin_edges, patches = plt.hist(data, histtype='step', label=label, bins=bins, color=color)
	bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	errors = np.sqrt(counts)
	plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# --------------------------------

# before cuts

signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

prev_bg = cosmic_bg_df.shape[0] + nu_bg_df.shape[0]

fig_save = directory+'background_type_over_track_len.png'
bins = np.linspace(0, 100, 50)
plt.clf()
fig, ax = plt.subplots()

plot_with_errors(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins, 'Cosmic Background', colors[1])
plot_with_errors(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins, 'Neutrino Background', colors[2])
plot_with_errors(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5])

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length', fontsize=16)
ax.set_ylabel('NC1p Candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#ax.text(0.58, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top')
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

print('Purity: ', signal_df.shape[0]/NC1p_df.shape[0])
print('Cosmics (ratio): ', cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print('Neutrino background (ratio): ', nu_bg_df.shape[0]/NC1p_df.shape[0])

# --------------------------------------

# barycenter

NC1p_df =  NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

fig_save = directory+'background_type_after_barycenter.png'
bins = np.linspace(0, 100, 50)
plt.clf()
fig, ax = plt.subplots()

plot_with_errors(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins, 'Cosmic Background', colors[1])
plot_with_errors(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins, 'Neutrino Background', colors[2])
plot_with_errors(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5])

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length', fontsize=16)
ax.set_ylabel('NC1p Candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#ax.text(0.58, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top')
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)


print('Purity: ', signal_df.shape[0]/NC1p_df.shape[0])
print('Cosmics (ratio): ', cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print('Neutrino background (ratio): ', nu_bg_df.shape[0]/NC1p_df.shape[0])
print('% of background removed: ', (prev_bg  - (cosmic_bg_df.shape[0] + nu_bg_df.shape[0]))/prev_bg)

