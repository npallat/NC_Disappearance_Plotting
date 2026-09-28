# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl
import matplotlib.font_manager as fm
from scipy.stats import beta

# ********* USER INPUT ************
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_ICARUS_SpringMC.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_new_ICARUS_SpringMC.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_short_ICARUS_SpringMC.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/PRN/'
#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'

expt = 'SBND' #'ICARUS'	#'SBND'

sample_POT = 5.30*(10**(20))
Run1_2_POT = 6.84*(10**(20))
POT = '6.84 x $10^{20}$ POT' # label
# *********************************

def plot_efficiency_old(passing, total_signal, bins, label, color, confidence=0.683):
    counts_pass, bin_edges = np.histogram(passing, bins=bins)
    counts_total, _ = np.histogram(total_signal, bins=bins)

    eff = np.where(counts_total > 0, counts_pass / counts_total, 0)

    alpha = 1 - confidence
    ci_low = np.where(
        counts_total > 0,
        beta.ppf(alpha / 2, counts_pass, counts_total - counts_pass + 1),
        0
    )
    ci_high = np.where(
        counts_total > 0,
        beta.ppf(1 - alpha / 2, counts_pass + 1, counts_total - counts_pass),
        0
    )

    # Handle edge cases where counts_pass == 0 or counts_pass == counts_total
    ci_low = np.where(counts_pass == 0, 0, ci_low)
    ci_high = np.where(counts_pass == counts_total, 1, ci_high)

    err_low = eff - ci_low
    err_high = ci_high - eff

    stair = plt.stairs(eff, bin_edges, label=label, color=color)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.errorbar(bin_centers, eff, yerr=[err_low, err_high], fmt='none', ecolor=color, capsize=3)

    return stair

def plot_efficiency(ax, passing, total_signal, bins, label, color, marker='o', confidence=0.683):
    counts_pass, bin_edges = np.histogram(passing, bins=bins)
    counts_total, _ = np.histogram(total_signal, bins=bins)

    eff = np.where(counts_total > 0, counts_pass / counts_total, 0)

    alpha = 1 - confidence
    ci_low = np.where(
        counts_total > 0,
        beta.ppf(alpha / 2, counts_pass, counts_total - counts_pass + 1),
        0
    )
    ci_high = np.where(
        counts_total > 0,
        beta.ppf(1 - alpha / 2, counts_pass + 1, counts_total - counts_pass),
        0
    )

    ci_low = np.where(counts_pass == 0, 0, ci_low)
    ci_high = np.where(counts_pass == counts_total, 1, ci_high)

    err_low = eff - ci_low
    err_high = ci_high - eff

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    handle = ax.errorbar(bin_centers, eff, yerr=[err_low, err_high],
                         fmt='-o', markersize=3, linewidth=1, color=color, capsize=3, elinewidth=1, label=label)
                         #fmt=f'{marker}-', color=color, capsize=3, label=label)

    return handle

if expt == 'ICARUS':
        x_min = -333.49
        x_max = 333.49
        y_min = -156.86
        y_max = 109.96
        z_min = -864.95
        z_max = 844.95
        # does not account for hanging cable
        # https://sbn-docdb.fnal.gov/cgi-bin/sso/RetrieveFile?docid=42799&filename=ICARUS_Numu_Disappearance_Technote_28Aug.pdf&version=1
elif expt == 'SBND':
        x_min = -190
        x_max = 190
        y_min = -190
        y_max = 190
        z_min = 10
        z_max = 500 #450
else:
        'Error with expt name. Exiting...'
        exit()

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
# --------------------------------

# remove duplicates
'''
df = (df.sort("rec.slc.reco.pfp.trk.len", descending=True, nulls_last=True).unique(
        #subset=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.evt", "rec.hdr.pot"],
        subset=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.pot"],
        keep="first"
    )
)
'''
total = df.shape[0]

fig_save = directory+'RecoEff_Efficiency_NuE.png'
bins = np.linspace(0,2,50)
plt.clf()
fig, ax = plt.subplots(figsize=(5.8, 4.8))

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_ke = df['rec.mc.nu.E']
#signal_line = plot_efficiency(ax, signal_ke, signal_ke, bins, 'True Signal', colors[0])

# [CRIT 0] in reconstruction
df = df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
reco_line = plot_efficiency(ax, df['rec.mc.nu.E'], signal_ke, bins, 'Reconstructed Signal', colors[1])
print(df.shape[0]/total)

# [CRIT 1] npfp <= 2
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
npfp_line = plot_efficiency(ax, df['rec.mc.nu.E'], signal_ke, bins, '1 Particle', colors[2])
print(df.shape[0]/total)

# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
#df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
#plot_efficiency(df['rec.mc.nu.prim.startE']-0.938, signal_ke, bins, 'Proton-like', colors[3])
print(df.shape[0]/total)

# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
pid_line = plot_efficiency(ax, df['rec.mc.nu.E'], signal_ke, bins, 'PID', colors[3])
print(df.shape[0]/total)

# [CRIT 4] All tracks contained in fiducial volume
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
cont_line = plot_efficiency(ax, df['rec.mc.nu.E'], signal_ke, bins, 'Contained', colors[4])
print(df.shape[0]/total)

# [CRIT 5] Flash matching
if expt == 'SBND':
        df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
        flash_line = plot_efficiency(ax, df['rec.mc.nu.E'], signal_ke, bins, 'Flash Match', colors[5])


ax.axhline(1.0, color=colors[0], linestyle='--', linewidth=1)
ax.set_ylim(0, 1.1)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('True Incoming Neutrino Energy [GeV]', fontsize=16)
ax.set_ylabel('Selection Efficiency', fontsize=16)
legend = ax.legend(fontsize=12, loc='best')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
width, height = fig.get_size_inches() * fig.dpi
print(width, height)
print('Plot saved as', fig_save)

