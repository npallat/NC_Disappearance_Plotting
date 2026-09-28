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

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'

expt = 'SBND' #'ICARUS'	#'SBND'

sample_POT = 5.30*(10**(20))
Run1_2_POT = 6.55*(10**(20))
POT = '6.55e20' # label
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
	#z_min = 10
	#z_max = 450
else:
	'Error with expt name. Exiting...'
	exit()

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
# --------------------------------

fig_save = directory+'Z_Fiducial_Volume_varations.png'
bins = np.linspace(0,1,25)
plt.clf()
fig, ax = plt.subplots()

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

total = df.shape[0]
df = df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
print(df.shape[0]/total)
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
print(df.shape[0]/total)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
print(df.shape[0]/total)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
print(df.shape[0]/total)
if expt == 'SBND':
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
print(df.shape[0]/total)


# z fiducial volume
z_pairs = [[0, 500], [10, 500], [10, 490], [10, 480], [10, 470], [10, 460], [10, 450]]
i = 0
while i < len(z_pairs):
	z_min = z_pairs[i][0]
	z_max = z_pairs[i][1]
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
	print(df.shape[0]/total)
	plot_with_errors(df['rec.mc.nu.prim.startE']-0.938, bins, str(z_min)+' < z < '+str(z_max), colors[i])
	i += 1

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Proton Kinetic Energy (GeV)', fontsize=16)
ax.set_ylabel('NC1p Candidates', fontsize=16)
legend = ax.legend(fontsize=14)
ax.add_artist(legend)
#if expt == 'SBND': ax.text(0.51, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top')
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

