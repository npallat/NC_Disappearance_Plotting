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
sbnd_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_ICARUS_SpringMC.csv", separator="\t")
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_new_ICARUS_SpringMC.csv", separator="\t")
icarus_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/efficiency/eff_short_ICARUS_SpringMC.csv", separator="\t")

#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'
# *********************************

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
                         fmt=f'-{marker}', markersize=3, linestyle='--', linewidth=1, color=color, capsize=3, elinewidth=1, label=label)
                         #fmt=f'{marker}-', color=color, capsize=3, label=label)

    return handle

icarus_df = icarus_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in icarus_df.columns])
sbnd_df = sbnd_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in sbnd_df.columns])
# --------------------------------

# remove duplicates
icarus_df = (icarus_df.sort("rec.slc.reco.pfp.trk.len", descending=True, nulls_last=True).unique(
        subset=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.evt", "rec.hdr.pot"],
        #subset=["rec.mc.nu.prim.G4ID", "rec.mc.nu.prim.pdg", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.pot"],
        keep="first"
    )
)
icarus_total = icarus_df.shape[0]
sbnd_total = sbnd_df.shape[0]

fig_save = directory+'Compare_RecoEff_Efficiency.png'
bins = np.linspace(0,1,25)
plt.clf()
fig, ax = plt.subplots(figsize=(8, 5))

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

sbnd_signal_ke = sbnd_df['rec.mc.nu.prim.startE'] - 0.938
plot_efficiency(ax, sbnd_signal_ke, sbnd_signal_ke, bins, 'SBND Signal', colors[0], marker='o')
icarus_signal_ke = icarus_df['rec.mc.nu.prim.startE'] - 0.938
plot_efficiency(ax, icarus_signal_ke, icarus_signal_ke, bins, 'ICARUS Signal', colors[0], marker='s')

# [CRIT 0] in reconstruction
sbnd_df = sbnd_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
plot_efficiency(ax, sbnd_df['rec.mc.nu.prim.startE']-0.938, sbnd_signal_ke, bins, 'SBND: In Reconstruction', colors[1], marker='o')
print(sbnd_df.shape[0]/sbnd_total)
icarus_df = icarus_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
plot_efficiency(ax, icarus_df['rec.mc.nu.prim.startE']-0.938, icarus_signal_ke, bins, 'ICARUS: In Reconstruction', colors[1], marker='s')
print(icarus_df.shape[0]/icarus_total, '\n')

# [CRIT 1] npfp <= 2
sbnd_df = sbnd_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
plot_efficiency(ax, sbnd_df['rec.mc.nu.prim.startE']-0.938, sbnd_signal_ke, bins, '1 Particle', colors[2], marker='o')
print(sbnd_df.shape[0]/sbnd_total)
icarus_df = icarus_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
plot_efficiency(ax, icarus_df['rec.mc.nu.prim.startE']-0.938, icarus_signal_ke, bins, '1 Particle', colors[2], marker='s')
print(icarus_df.shape[0]/icarus_total, '\n')

# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
sbnd_df = sbnd_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
#plot_efficiency(df['rec.mc.nu.prim.startE']-0.938, signal_ke, bins, 'Proton-like', colors[3])
print(sbnd_df.shape[0]/sbnd_total)
print(icarus_df.shape[0]/icarus_total, '\n')

# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
sbnd_df = sbnd_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
plot_efficiency(ax, sbnd_df['rec.mc.nu.prim.startE']-0.938, sbnd_signal_ke, bins, 'PID', colors[3], marker='o')
print(sbnd_df.shape[0]/sbnd_total)
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
plot_efficiency(ax, icarus_df['rec.mc.nu.prim.startE']-0.938, icarus_signal_ke, bins, 'PID', colors[3], marker='s')
print(icarus_df.shape[0]/icarus_total, '\n')

# [CRIT 4] All tracks contained in fiducial volume
x_min = -190
x_max = 190
y_min = -190
y_max = 190
z_min = 10
z_max = 500 #450
sbnd_df = sbnd_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
sbnd_df = sbnd_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
plot_efficiency(ax, sbnd_df['rec.mc.nu.prim.startE']-0.938, sbnd_signal_ke, bins, 'Contained', colors[4], marker='o')
print(sbnd_df.shape[0]/sbnd_total)

x_min = -333.49
x_max = 333.49
y_min = -156.86
y_max = 109.96
z_min = -864.95
z_max = 844.95
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
icarus_df = icarus_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
plot_efficiency(ax, icarus_df['rec.mc.nu.prim.startE']-0.938, icarus_signal_ke, bins, 'Contained', colors[4], marker='s')
print(icarus_df.shape[0]/icarus_total, '\n')

# [CRIT 5] Flash matching
sbnd_df = sbnd_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
plot_efficiency(ax, sbnd_df['rec.mc.nu.prim.startE']-0.938, sbnd_signal_ke, bins, 'Flash Match', colors[5], marker='o')
print(sbnd_df.shape[0]/sbnd_total)
icarus_df = icarus_df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 
icarus_df = icarus_df.filter(pl.col('rec.slc.barycenterFM.radius') < 110) 
plot_efficiency(ax, icarus_df['rec.mc.nu.prim.startE']-0.938, icarus_signal_ke, bins, 'Flash Match', colors[5], marker='s')
print(icarus_df.shape[0]/icarus_total, '\n')

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Proton Kinetic Energy (GeV)', fontsize=16)
ax.set_ylabel('Selection Efficiency', fontsize=16)
legend = ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
#legend = ax.legend(fontsize=9)
#ax.add_artist(legend)
#if expt == 'SBND': ax.text(0.56, 0.91, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top')
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

