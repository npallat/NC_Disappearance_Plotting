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
df = pl.read_csv("/pnfs/sbnd/persistent/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/SBND_Gen2CV_Sample/'
save_df = directory+'SBNDGen2CV_cheated_signal_mult_hits.csv'

expt = 'SBND'

cheated = True

bin_edges = np.linspace(0.0, 0.5, 13)  # 12 bins, 0 to 0.5 GeV
# full 
#pct_increase = [0.00, 0.06, 0.23, 0.12, 0.06, 0.10, 0.10, 0.03, 0.07, 0.00, 0.00, 0.03]
# split tracks
#pct_increase = [0.00, 0.00, 0.00, 0.03, 0.04, 0.02, 0.05, 0.03, 0.03, 0.00, 0.00, 0.03]
# cosmics
# pct_increase = [0.00, 0.05, 0.22, 0.09, 0.03, 0.08, 0.06, 0.00, 0.04, 0.00, 0.00, 0.00]
# unreasonable
#pct_increase = [0.00, 0.05, 0.22, 0.09, 0.03, 0.08, 0.06, 0.00, 0.04, 0.00, 0.00, 0.00]
# impossible
#pct_increase = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# mult. plane reco improvement
#pct_increase = [0.00, 0.04, 0.23, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
# mult. hits reco improvement
pct_increase = [0.00, 0.18, 0.29, 0.21, 0.00, 0.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

# *********************************

sample_POT = 4.8603*(10**(20))
Run1_2_POT = 6.84*(10**(20))
POT = '6.84 x $10^{20}$ POT' # label

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

if expt == 'SBND':
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

scale = Run1_2_POT/sample_POT

df = df.filter(pl.col('signal.NC1p').is_not_null())
total = df.shape[0]
# copy of signal df
signal_df = df

# [CRIT 0] in reconstruction
df = df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())
# [CRIT 1] npfp <= 2
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
# [CRIT 4] All tracks contained in fiducial volume
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
# [CRIT 5] Flash matching
df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

if cheated == False:
	df.write_csv(save_df, separator="\t")
	print("Saved", save_df)
	exit()

# --------------------------------

# failed events
ID_COLS = ['entry', 'rec.hdr.run', 'rec.hdr.subrun', 'rec.mc.nu.E', 'rec.slc.reco.pfp.trk.truth.p.interaction_id']
failed_df = signal_df.join(df.select(ID_COLS), on=ID_COLS, how='anti')

#TRUE_KE_COL = ['rec.mc.nu.prim.startE']-0.938 

rng_seed = 42
extra_frames = []

for i in range(12):
    lo, hi = bin_edges[i], bin_edges[i + 1]

    n_signal_bin = signal_df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi)).height

    n_passed_bin = df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi)).height

    eff = n_passed_bin / n_signal_bin if n_signal_bin > 0 else float('nan')

    target_eff = min(eff + pct_increase[i], 1.0)
    n_to_add = round(target_eff * n_signal_bin) - n_passed_bin
    n_to_add = max(n_to_add, 0)

    pool_bin = failed_df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi))
    n_available = pool_bin.height

    if n_to_add > n_available:
        print(f"[bin {i}] wanted {n_to_add}, only {n_available} failed events available — adding all")
        n_to_add = n_available

    if n_to_add > 0:
        extra_frames.append(pool_bin.sample(n=n_to_add, seed=rng_seed + i))

extra_df = pl.concat(extra_frames) if extra_frames else df.clear()
cheated_df = pl.concat([df, extra_df])

# ----------------
print(f"{'Bin (GeV)':>15} | {'N_signal':>8} | {'N_passed':>8} | {'Eff':>6} | {'N_cheat':>8} | {'CheatEff':>8}")
print("-" * 70)

for i in range(12):
    lo, hi = bin_edges[i], bin_edges[i + 1]

    n_signal_bin = signal_df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi)).height

    n_passed_bin = df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi)).height

    n_cheated_bin = cheated_df.filter((pl.col('rec.mc.nu.prim.startE')-0.938 >= lo) & (pl.col('rec.mc.nu.prim.startE')-0.938 < hi)).height

    eff = n_passed_bin / n_signal_bin if n_signal_bin > 0 else float('nan')
    cheat_eff = n_cheated_bin / n_signal_bin if n_signal_bin > 0 else float('nan')

    print(f"[{lo:.3f}, {hi:.3f})  | {n_signal_bin:>8} | {n_passed_bin:>8} | {eff:6.3f} | {n_cheated_bin:>8} | {cheat_eff:8.3f}")
# ----------------

cheated_df.write_csv(save_df, separator="\t")
print("Saved", save_df)

