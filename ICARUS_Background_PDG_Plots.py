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
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")
#icarus_df = pl.read_csv("/pnfs/sbnd/persistent/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")
icarus_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC_Full/combined/ICARUSSpringMC_with_background.csv", separator="\t")

#directory = '/pnfs/sbnd/persistent/users/npallat/NC_disappearance/ICARUS_Spring26/plots/NewPerspectives/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC_Full/plots/'

bins = np.linspace(0,100,50)

sample_POT = 4.8603*(10**(20))
Run1_2_POT = 6.84*(10**(20))
POT = '6.84 x $10^{20}$ POT' # label
scale = Run1_2_POT/sample_POT
# *********************************

icarus_df = icarus_df.select(['signal.NC1p', 'rec.slc.reco.pfp.trk.len', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.npfp', 
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
			'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon', 'rec.slc.reco.pfp.trk.start.x',
			'rec.slc.reco.pfp.trk.start.y', 'rec.slc.reco.pfp.trk.start.z', 'rec.slc.reco.pfp.trk.end.x', 'rec.slc.reco.pfp.trk.end.y', 'rec.slc.reco.pfp.trk.end.z', 
			'rec.mc.nu.prim.startE', 'rec.mc.nu.prim.G4ID', 'rec.mc.nu.prim.pdg', 'rec.hdr.run', 'rec.hdr.subrun', 'rec.hdr.evt', 'rec.hdr.pot', 
			'rec.slc.reco.pfp.trk.truth.p.interaction_id', 'rec.slc.is_clear_cosmic', 'rec.slc.barycenterFM.flashTime', 'rec.slc.barycenterFM.radius'])

icarus_df = icarus_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in icarus_df.columns])

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

# [CRIT 0] in reconstruction
icarus_df = icarus_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())

# [CRIT 1] npfp <= 2
icarus_df = icarus_df.filter(pl.col('rec.slc.reco.npfp') <= 2)

# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
#icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))

# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))

# [CRIT 4] All tracks contained in fiducial volume
x_min = -333.49
x_max = 333.49
y_min = -156.86
y_max = 109.96
z_min = -864.95
z_max = 844.95
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))

# [CRIT 5] Flash matching
icarus_df = icarus_df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 
icarus_df = icarus_df.filter(pl.col('rec.slc.barycenterFM.radius') < 110) 

# Split into Signal and Background
signal_df = icarus_df.filter(pl.col('signal.NC1p') == 1)
bg_df = icarus_df.filter(pl.col('signal.NC1p').is_null())

df_total = icarus_df.shape[0]
print('% signal: ', signal_df.shape[0]/df_total)
print('% protons: ', icarus_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/df_total)
print('% muons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/df_total)
print('% ch pions: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211).shape[0]/df_total)
print('% nuclear: ', bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') >= 1000000).shape[0]/df_total)
print('% photons: ', bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22).shape[0]/df_total)
print('% electrons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/df_total)
print('% cosmics: ', bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') < 0).shape[0]/df_total)
print('pdg codes: ', np.unique(bg_df['rec.slc.reco.pfp.trk.truth.p.pdg']))

# dataframes: signal, signal duplicates (from split tracks, shorter), background
# remove duplicates
'''
signal_nodup_df = (signal_df.sort("rec.slc.reco.pfp.trk.len", descending=True, nulls_last=True).unique(
        subset=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.evt", "rec.hdr.pot"],
        keep="first"
    )
)
signal_split_df = signal_df.join(signal_nodup_df, on=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.evt", "rec.hdr.pot", "rec.slc.reco.pfp.trk.len"], how='anti')
bg_df = df.filter(pl.col('signal.NC1p').is_null())
'''

# plot 
'''
fig_save = directory+'ICARUS_background_pdg.png'
plt.clf()
fig, ax = plt.subplots()
stair_other, bottom = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Other', colors[4], scale=scale)
stair_signal, bottom = plot_stacked(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale, bottom=bottom)
stair_protons, total = plot_stacked(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Non-Signal Protons', colors[2], scale=scale, bottom=bottom)
band = plot_error_band(total, bins, scale=scale)
ax.set_xlim(0, 100)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length [cm]', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(handles=[stair_protons, stair_signal, stair_other, band], fontsize=12)
ax.add_artist(legend)
ax.text(0.42, 0.53, f'SBN Analysis in Progress', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
ax.text(0.42, 0.47, f'ICARUS Simulation', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
ax.text(0.42, 0.41, f'{POT}', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
print('Plot saved as', fig_save)
'''
