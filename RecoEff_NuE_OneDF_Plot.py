# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl
import matplotlib.font_manager as fm

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

def plot_with_errors(data, bins, label, color, scale=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts)

    if scale is not None:
        counts = counts * scale
        errors = errors * scale

    stair = plt.stairs(counts, bin_edges, label=label, color=color)
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)

    return stair

if expt == 'ICARUS':
	x_min = -348.5
	x_max = 348.5
	y_min = -171.9
	y_max = 125.0
	z_min = -885
	z_max = 885
	#x_min = -333.49
	#x_max = 333.49
	#y_min = -156.86
	#y_max = 109.96
	#z_min = -864.95
	#z_max = 844.95
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
# scale
scale = Run1_2_POT/sample_POT
if expt == 'ICARUS': scale = 1

total = df.shape[0]

fig_save = directory+'NuE.png'
bins = np.linspace(0,2,50)
plt.clf()
fig, ax = plt.subplots()

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

signal_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'True Signal', colors[0], scale=scale)

# [CRIT 0] in reconstruction
df = df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
reco_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'Reconstructed Signal', colors[1], scale=scale)
print(df.shape[0]/total)

# [CRIT 1] npfp <= 2
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
npfp_line = plot_with_errors(df['rec.mc.nu.E'], bins, '1 Particle', colors[2], scale=scale)
print(df.shape[0]/total)

# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
#df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
#plot_with_errors(df['rec.mc.nu.prim.startE']-0.938, bins, 'Proton-like', colors[3])
print(df.shape[0]/total)

# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
pid_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'PID', colors[3], scale=scale)
print(df.shape[0]/total)

# [CRIT 4] All tracks contained in fiducial volume
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
cont_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'Contained', colors[4], scale=scale)
print(df.shape[0]/total)

# [CRIT 5] Flash matching
if expt == 'SBND':
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
	flash_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'Flash Match', colors[5], scale=scale)
	print(df.shape[0]/total)
else:
	df = df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 
	df = df.filter(pl.col('rec.slc.barycenterFM.radius') < 110) 
	cosmic_line = plot_with_errors(df['rec.mc.nu.E'], bins, 'Cosmic Removal', colors[5], scale=scale)
	print(df.shape[0]/total)

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('True Incoming Neutrino Energy [GeV]', fontsize=16)
ax.set_ylabel('NC1p Candidates', fontsize=16)
legend = ax.legend(fontsize=12, loc='best')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
width, height = fig.get_size_inches() * fig.dpi
print(width, height)
print('Plot saved as', fig_save)

