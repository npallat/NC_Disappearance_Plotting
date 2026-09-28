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
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/NC1p_curr_selection_no_bary.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/merge_fix/'
# *********************************

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
# merge fix
left_indicator  = "rec.mc.nu.E"        # null iff row came from right only
right_indicator = "rec.slc.reco.pfp.trk.truth.p.interaction_id"   # null iff row came from left only

left_keys  = ["rec.mc.nu.prim.G4ID",               "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.pot", "rec.hdr.triggerinfo.passed_trigger"]
right_keys = ["rec.slc.reco.pfp.trk.truth.p.G4ID", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.pot", "rec.hdr.triggerinfo.passed_trigger"]

# Rows matched on all 5 keys
matched = df.filter(pl.col(left_indicator).is_not_null() & pl.col(right_indicator).is_not_null())

# Left only rows: no right side match (right indicator is null)
left_only = df.filter(pl.col(right_indicator).is_null())

# Right only rows: no left side match (left indicator is null)
right_only = df.filter(pl.col(left_indicator).is_null())

# Re-join the orphaned fragments on the correct keys
shared_cols = ["entry", "rec.hdr.run", "rec.hdr.subrun", "rec.hdr.triggerinfo.passed_trigger", "rec.hdr.pot"]

recovered = left_only.join(
    right_only,
    left_on=left_keys,
    right_on=right_keys,
    how="full"
).with_columns([
    pl.coalesce([col, col + "_right"]).alias(col)
    for col in shared_cols
]).drop([col + "_right" for col in shared_cols])

df = pl.concat([matched, recovered], how="diagonal_relaxed")

# Combine everything
df = pl.concat([matched, recovered], how="diagonal_relaxed")
# *********************************

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())

# numbers
total = signal_df.shape[0] + bg_df.shape[0]
print("Purity: ", signal_df.shape[0]/total)
print("All Protons: ", (signal_df.shape[0] + bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0])/total)
print("Background Protons: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/total)
print("Muons: ", bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/total)
print("Photons: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22).shape[0]/total)
print("Electrons: ", bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/total)
print("Other: ", (bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
			)).shape[0]/total)


# plot 

fig_save = directory+'background_pdg_over_track_len.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Signal Protons')
plt.hist(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Protons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 13) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -13))['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Muons')
plt.hist(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Photons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 11) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -11))['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Electrons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
			)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Other')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Track Length', fontsize=16)
plt.ylabel('NC1p candidates', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

# --------------------------------------

# Cosmic removal cuts
#signal_df =  signal_df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
#signal_df =  signal_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1)
signal_df =  signal_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
#bg_df =  bg_df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
#bg_df =  bg_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1)
bg_df =  bg_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

# remake plot
fig_save = directory+'background_pdg_after_cosmic_cuts.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Signal Protons')
plt.hist(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Protons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 13) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -13))['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Muons')
plt.hist(bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Photons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 11) | (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -11))['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Electrons')
plt.hist(bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
			)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Other')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Track Length', fontsize=16)
plt.ylabel('NC1p candidates', fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(fig_save, bbox_inches='tight')
print('Plot saved as', fig_save)

# --------------------------------------

# numbers

total = signal_df.shape[0] + bg_df.shape[0]
print("Purity: ", signal_df.shape[0]/total)
print("All Protons: ", (signal_df.shape[0] + bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0])/total)
print("Background Protons: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/total)
print("Muons: ", bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]/total)
print("Photons: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22).shape[0]/total)
print("Electrons: ", bg_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11).shape[0]/total)
print("Other: ", (bg_df.filter((pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -11) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -13) &
			(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212)
			)).shape[0]/total)


print("\nBackgrounds: ")
bg_total = bg_df.shape[0]
print("NC: ", bg_df.filter(pl.col('rec.slc.truth.isnc') == 1).shape[0]/bg_total)
print("CC: ", bg_df.filter(pl.col('rec.slc.truth.isnc') != 1).shape[0]/bg_total)
print("Null NC: ", bg_df.filter(pl.col('rec.slc.truth.isnc').is_null()).shape[0]/bg_total)
print("In Cryostat: ", bg_df.filter(pl.col('rec.mc.nu.cryostat') != -1).shape[0]/bg_total)
print("Outside Cryostat: ", bg_df.filter(pl.col('rec.mc.nu.cryostat') == -1).shape[0]/bg_total)
print("Null Cryostat: ", bg_df.filter(pl.col('rec.mc.nu.cryostat').is_null()).shape[0]/bg_total)
print("Cosmic: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/bg_total)
print("Neutrino: ", bg_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1).shape[0]/bg_total)
print("N prim == 3: ", bg_df.filter(pl.col('rec.mc.nu.nprim') == 3).shape[0]/bg_total)
print("N prim > 3: ", bg_df.filter(pl.col('rec.mc.nu.nprim') > 3).shape[0]/bg_total)
print(np.unique(bg_df['rec.mc.nu.nproton']))
'''
print("Neutrons > 0: ", bg_df.filter(pl.col('rec.mc.nu.nneutron') > 0).shape[0]/bg_total)
print("No Neutrons: ", bg_df.filter(pl.col('rec.mc.nu.nneutron') == 0).shape[0]/bg_total)
print("Protons > 0: ", bg_df.filter(pl.col('rec.mc.nu.nproton') > 0).shape[0]/bg_total)
print("No Protons: ", bg_df.filter(pl.col('rec.mc.nu.nproton') == 0).shape[0]/bg_total)
print("Pions > 0: ", bg_df.filter((pl.col('rec.mc.nu.npiplus') + pl.col('rec.mc.nu.npiminus') + pl.col('rec.mc.nu.npizero')) > 0).shape[0]/bg_total)
print("No Pions: ", bg_df.filter((pl.col('rec.mc.nu.npiplus') + pl.col('rec.mc.nu.npiminus') + pl.col('rec.mc.nu.npizero')) == 0).shape[0]/bg_total)
'''

'''
    'rec.mc.nu.nprim',

'''
