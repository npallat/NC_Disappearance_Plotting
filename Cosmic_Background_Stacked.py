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
#df = pl.read_csv("/exp/sbnd/app/users/npallat/NC_disappearance/csv_files/polars/parallel/gen2/with_bg_bary/NC1p_reco_V5_full.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/Fran/V2/FranSelection_Gen2_2026_OnlyNC1pTruth.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/NC1p_curr_selection_no_bary.csv", separator="\t")

#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")

#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/June26CM/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/June26CM/'
#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Fran_Sample/plots/'
sample_POT = 5.30*(10**(20))
Run1_2_POT = 6.84*(10**(20))
POT = '6.84 x $10^{20}$ POT' # label
# *********************************

#scale = Run1_2_POT/sample_POT
scale = 1

#df = df.select(['signal.NC1p', 'rec.slc.reco.pfp.trk.len', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.npfp', 
#			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
#			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon', 'rec.slc.reco.pfp.trk.start.x',
#			'rec.slc.reco.pfp.trk.start.y', 'rec.slc.reco.pfp.trk.start.z', 'rec.slc.reco.pfp.trk.end.x', 'rec.slc.reco.pfp.trk.end.y', 'rec.slc.reco.pfp.trk.end.z', 
#			'rec.slc.barycenterFM.score', 'rec.mc.nu.prim.startE', 'rec.slc.reco.pfp.trk.truth.p.interaction_id'])

df = df.select(['signal.NC1p', 'rec.slc.reco.pfp.trk.len', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.npfp', 
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon', 'rec.slc.reco.pfp.trk.start.x',
			'rec.slc.reco.pfp.trk.start.y', 'rec.slc.reco.pfp.trk.start.z', 'rec.slc.reco.pfp.trk.end.x', 'rec.slc.reco.pfp.trk.end.y', 'rec.slc.reco.pfp.trk.end.z', 
			'rec.slc.barycenterFM.flashTime', 'rec.slc.barycenterFM.radius', 'rec.mc.nu.prim.startE', 'rec.slc.reco.pfp.trk.truth.p.interaction_id'])

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
print(df.filter(pl.col('signal.NC1p') == 1).shape[0]/df.shape[0])
#df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
print(df.filter(pl.col('signal.NC1p') == 1).shape[0]/df.shape[0])

# SBND
#df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 500))
#df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 500))

# ICARUS
x_min = -333.49
x_max = 333.49
y_min = -156.86
y_max = 109.96
z_min = -864.95
z_max = 844.95
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
print(df.filter(pl.col('signal.NC1p') == 1).shape[0]/df.shape[0])
# *********************************

def plot_with_errors(data, bins, label, color, scale=None, bottom=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    if scale is not None:
        counts = counts * scale
        errors = errors * scale

    top = counts + (bottom if bottom is not None else 0)

    stair = plt.stairs(top, bin_edges, label=label, color=color, fill=True,
                       baseline=bottom if bottom is not None else 0)
    plt.errorbar(bin_centers, top, yerr=errors, fmt='none', ecolor='black', capsize=3)

    return stair, top

def plot_stacked(data, bins, label, color, scale=None, bottom=None):
    counts, bin_edges = np.histogram(data, bins=bins)

    if scale is not None:
        counts = counts * scale

    top = counts + (bottom if bottom is not None else 0)

    stair = plt.stairs(top, bin_edges, label=label, color=color, fill=True,
                       baseline=bottom if bottom is not None else 0)

    return stair, top

def plot_error_band(total_counts, bins, scale=None):
    counts_unscaled = total_counts / scale if scale is not None else total_counts
    errors = np.sqrt(counts_unscaled)
    if scale is not None:
        errors = errors * scale

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    band = ax.fill_between(bin_centers, total_counts - errors, total_counts + errors,
                           alpha=0.3, color='gray', step='mid',
                           label='Stat. Uncertainty')
    return band

# --------------------------------
colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())

# Cosmic removal cuts
#signal_df =  signal_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
#bg_df =  bg_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

signal_df = signal_df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 
signal_df = signal_df.filter(pl.col('rec.slc.barycenterFM.radius') < 110) 
bg_df = bg_df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 
bg_df = bg_df.filter(pl.col('rec.slc.barycenterFM.radius') < 110) 
print(signal_df.shape[0]/(signal_df.shape[0]+bg_df.shape[0]))

# --------------------------------------

# plot
'''
fig_save = directory+'cosmic_background_up_to_muon_chi2.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()
stair_signal, bottom = plot_stacked(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5])
stair_nonsignal, bottom = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') >= 0)['rec.slc.reco.pfp.trk.len'], bins, 'Neutrino-induced Background', colors[2], bottom=bottom)
stair_cosmic, total = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') < 0)['rec.slc.reco.pfp.trk.len'], bins, 'Cosmic-induced', colors[1], bottom=bottom)
band = plot_error_band(total, bins)
ax.set_xlim(0, 100)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length [cm]', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(handles=[stair_cosmic, stair_nonsignal, stair_signal, band], fontsize=14)
ax.add_artist(legend)
ax.text(0.42, 0.47, f'ICARUS Simulation', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
print('Plot saved as', fig_save)
'''
print(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)).shape[0])
print(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') >= 0)).shape[0])
print(signal_df.shape[0])
print(bg_df.shape[0])
exit()

# plot
fig_save = directory+'background_pdg_origin_after_cosmic_cuts.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()
#stair_other, bottom = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Other', colors[4], scale=scale)
#stair_signal, bottom = plot_stacked(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale, bottom=bottom)
#stair_nonsignal, total = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Non-Signal Protons', colors[2], scale=scale, bottom=bottom)

stair_signal, bottom = plot_stacked(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale)
stair_nonsignal, bottom = plot_stacked(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212) & (pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') >= 0))['rec.slc.reco.pfp.trk.len'], bins, 'Neutrino-induced Non-Signal Protons', colors[2], scale=scale, bottom=bottom)
stair_cosmic_nonsignal, bottom = plot_stacked(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212) & (pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') < 0))['rec.slc.reco.pfp.trk.len'], bins, 'Cosmic-induced Non-Signal Protons', colors[7], scale=scale, bottom=bottom)
stair_other_cosmic, bottom = plot_stacked(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212) & (pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') < 0))['rec.slc.reco.pfp.trk.len'], bins, 'Other Cosmic-induced', colors[1], scale=scale, bottom=bottom)
stair_other_neutrino, total = plot_stacked(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212) & (pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') >= 0))['rec.slc.reco.pfp.trk.len'], bins, 'Other Neutrino-induced', colors[4], scale=scale, bottom=bottom)

band = plot_error_band(total, bins, scale=scale)
ax.set_xlim(0, 100)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length [cm]', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(handles=[stair_other_neutrino, stair_other_cosmic, stair_cosmic_nonsignal, stair_nonsignal, stair_signal, band], fontsize=12)
ax.add_artist(legend)
ax.text(0.42, 0.47, f'ICARUS Simulation', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
print('Plot saved as', fig_save)

