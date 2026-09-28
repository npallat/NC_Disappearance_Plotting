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
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'

#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/NC1p_curr_selection.csv", separator="\t")
#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'

#var_to_plot = 'rec.slc.reco.pfp.trk.len'
var_to_plot = 'rec.slc.reco.nstub'
#bins = np.linspace(0,100,50)
bins = np.linspace(0,10,10)
# *********************************

df = df.select([var_to_plot, 'signal.NC1p'])#, 'rec.slc.reco.pfp.trk.len', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.npfp', 
			#'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
			#'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon', 'rec.slc.reco.pfp.trk.start.x',
			#'rec.slc.reco.pfp.trk.start.y', 'rec.slc.reco.pfp.trk.start.z', 'rec.slc.reco.pfp.trk.end.x', 'rec.slc.reco.pfp.trk.end.y', 'rec.slc.reco.pfp.trk.end.z', 
			#'rec.mc.nu.prim.startE'])

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])

def plot_with_errors(data, bins, label, color):
	counts, bin_edges, patches = plt.hist(data, histtype='step', label=label, bins=bins, color=color)
	bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	errors = np.sqrt(counts)
	plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)

# --------------------------------
colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())

# plot 

fig_save = directory+'signal_over_nstub.png'
plt.clf()
fig, ax = plt.subplots()
plot_with_errors(signal_df[var_to_plot], bins, 'Signal Protons', colors[5])
plot_with_errors(bg_df[var_to_plot], bins, 'Background', colors[2])
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel(var_to_plot, fontsize=16)
ax.set_ylabel('NC1p candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#ax.text(0.51, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top',)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

