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
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/PRN/'
#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Fran_Sample/plots/'
sample_POT = 5.30*(10**(20))
Run1_2_POT = 6.84*(10**(20))
POT = '6.84 x $10^{20}$ POT' # label
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
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 500))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 500))

# *********************************

def plot_with_errors(data, bins, label, color, scale=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    if scale is not None:
        counts = counts * scale
        errors = errors * scale

    plt.stairs(counts, bin_edges, label=label, color=color)
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)

# --------------------------------
colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())
'''
# plot 
fig_save = directory+'background_pdg_over_track_len.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Other Protons', colors[2], scale=scale)
plot_with_errors(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale)
#plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 13) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -13))['rec.slc.reco.pfp.trk.len'], bins, 'Muons', colors[1], scale=scale)
#plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)['rec.slc.reco.pfp.trk.len'], bins, 'Photons', colors[0], scale=scale)
#plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 11) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -11))['rec.slc.reco.pfp.trk.len'], bins, 'Electrons', colors[3], scale=scale)
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Other', colors[4], scale=scale)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#ax.text(0.51, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

print('Signal: ', signal_df.shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Electrons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Photons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 22).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Muons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Protons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 2212).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Charged Pions: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Nuclear Isotopes: ',bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') > 1000000).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))

print('\nElectrons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/(bg_df.shape[0]))
print('Photons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 22).shape[0]/(bg_df.shape[0]))
print('Muons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/(bg_df.shape[0]))
print('Protons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 2212).shape[0]/(bg_df.shape[0]))
print('Charged Pions: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211).shape[0]/(bg_df.shape[0]))
print('Nuclear Isotopes: ',bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') > 1000000).shape[0]/(bg_df.shape[0]))
'''

# --------------------------------------

# Cosmic removal cuts
signal_df =  signal_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
bg_df =  bg_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

# remake plot
fig_save = directory+'background_pdg_after_cosmic_cuts.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Non-Signal Protons', colors[2], scale=scale)
plot_with_errors(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale)
#plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 13) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -13))['rec.slc.reco.pfp.trk.len'], bins, 'Muons', colors[1], scale=scale)
#plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)['rec.slc.reco.pfp.trk.len'], bins, 'Photons', colors[0], scale=scale)
#plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 11) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -11))['rec.slc.reco.pfp.trk.len'], bins, 'Electrons', colors[3], scale=scale)
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Other', colors[4], scale=scale)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length [cm]', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
ax.text(0.46, 0.68, f'SBN Analysis in Progress', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
ax.text(0.46, 0.62, f'SBND Simulation', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
ax.text(0.46, 0.56, f'{POT}', transform=ax.transAxes, fontsize=16, color='gray', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
print('Plot saved as', fig_save)
'''
# log plot
fig_save = directory+'background_pdg_after_cosmic_cuts_log.png'
bins = np.linspace(0,100,50)
plt.clf()
fig, ax = plt.subplots()
plot_with_errors(signal_df['rec.slc.reco.pfp.trk.len'], bins, 'Signal Protons', colors[5], scale=scale)
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins, 'Protons', colors[2], scale=scale)
plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 13) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -13))['rec.slc.reco.pfp.trk.len'], bins, 'Muons', colors[1], scale=scale)
plot_with_errors(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)['rec.slc.reco.pfp.trk.len'], bins, 'Photons', colors[0], scale=scale)
plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 11) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -11))['rec.slc.reco.pfp.trk.len'], bins, 'Electrons', colors[3], scale=scale)
plot_with_errors(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
                        )['rec.slc.reco.pfp.trk.len'], bins, 'Other', colors[4], scale=scale)
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Track Length', fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(fontsize=10)
ax.add_artist(legend)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

print(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
                        (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
                        ).shape[0])
print('Signal: ', signal_df.shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Muons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Photons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 22).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Electrons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Protons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 2212).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Charged Pions: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))
print('Nuclear Isotopes: ',bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') > 1000000).shape[0]/(signal_df.shape[0]+bg_df.shape[0]))

print('\nMuons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/(bg_df.shape[0]))
print('Photons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 22).shape[0]/(bg_df.shape[0]))
print('Electrons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/(bg_df.shape[0]))
print('Protons: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 2212).shape[0]/(bg_df.shape[0]))
print('Charged Pions: ', bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211).shape[0]/(bg_df.shape[0]))
print('Nuclear Isotopes: ',bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') > 1000000).shape[0]/(bg_df.shape[0]))
'''
