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
NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/cosmic_bg/'
# *********************************

NC1p_df = NC1p_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in NC1p_df.columns])
NC1p_df = NC1p_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 450))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 450))

# *********************************
def bary_cut(df, cut=0.02):
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > cut)
	return df

def bary_crtveto_cut(df, cut=0.02):
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > cut)
	df = df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
	return df

def bary_crtveto_cosmic_cut(df, cut=0.02):
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > cut)
	df = df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
	df = df.filter(pl.col('rec.slc.is_clear_cosmic') != 1)
	return df

def crtveto_cut(df):
	df = df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
	return df

def cosmic_cut(df):
	df = df.filter(pl.col('rec.slc.is_clear_cosmic') != 1)
	return df

# *********************************
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

# all cuts

fig_save = directory+'background_type_all_cuts.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Cosmic background', color='tab:blue')
plt.hist(bary_cut(cosmic_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Flash Match', color='#3EA4CD')
plt.hist(bary_crtveto_cut(cosmic_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ CRT Veto', color='#5ED2E6')
plt.hist(bary_crtveto_cosmic_cut(cosmic_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Clear Cosmic', color="#7DFFFF")
plt.hist(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Neutrino background', color='tab:orange')
plt.hist(bary_cut(nu_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Flash Match', color='#FF972E')
plt.hist(bary_crtveto_cut(nu_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ CRT Veto', color='#FFB04E')
plt.hist(bary_crtveto_cosmic_cut(nu_bg_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Clear Cosmic', color='#FFC86E')
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='Signal protons', color='tab:green')
plt.hist(bary_cut(signal_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Flash Match', color='#57C84D')
plt.hist(bary_crtveto_cut(signal_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ CRT Veto', color='#83D475')
plt.hist(bary_crtveto_cosmic_cut(signal_df)['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='+ Clear Cosmic', color='#ABE098')
plt.xlabel('Track Length')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

# --------------------------------------

# before cuts

with open(directory+"Cosmic_Background_Info.txt", "w") as file:
	file.write("-- Before Cosmic Removal --")
	file.write('\nPurity: ')
	file.write(str(signal_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nCosmics (ratio): ')
	file.write(str(cosmic_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nNeutrino Background (ratio): ')
	file.write(str(nu_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nRatio of selection = protons: ')
	file.write(str(NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0]))
	file.write('\nAbsolute # signal: ')
	file.write(str(signal_df.shape[0]))
	file.write('\nAbsolute # Cosmics: ')
	file.write(str(cosmic_bg_df.shape[0]))
	file.write('\nAbsolute # Neutrino Background: ')
	file.write(str(nu_bg_df.shape[0]))


fig_save = directory+'background_type_over_track_len.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Track Length')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

# --------------------------------------

# barycenter

NC1p_df =  NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

fig_save = directory+'background_type_after_barycenter.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Track Length')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)


with open(directory+"Cosmic_Background_Info.txt", "a") as file:
	file.write("\n\n-- After Bary Flash Matching  --")
	file.write('\nPurity: ')
	file.write(str(signal_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nCosmics (ratio): ')
	file.write(str(cosmic_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nNeutrino Background (ratio): ')
	file.write(str(nu_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nRatio of selection = protons: ')
	file.write(str(NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0]))
	file.write('\nAbsolute # signal: ')
	file.write(str(signal_df.shape[0]))
	file.write('\nAbsolute # Cosmics: ')
	file.write(str(cosmic_bg_df.shape[0]))
	file.write('\nAbsolute # Neutrino Background: ')
	file.write(str(nu_bg_df.shape[0]))

# --------------------------------------

# crt veto
NC1p_df =  NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1);
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

fig_save = directory+'background_type_after_crtveto.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Track Length')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

with open(directory+"Cosmic_Background_Info.txt", "a") as file:
	file.write("\n\n-- After CRT Veto  --")
	file.write('\nPurity: ')
	file.write(str(signal_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nCosmics (ratio): ')
	file.write(str(cosmic_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nNeutrino Background (ratio): ')
	file.write(str(nu_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nRatio of selection = protons: ')
	file.write(str(NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0]))
	file.write('\nAbsolute # signal: ')
	file.write(str(signal_df.shape[0]))
	file.write('\nAbsolute # Cosmics: ')
	file.write(str(cosmic_bg_df.shape[0]))
	file.write('\nAbsolute # Neutrino Background: ')
	file.write(str(nu_bg_df.shape[0]))

# --------------------------------------

# clear cosmic
NC1p_df =  NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1);
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

fig_save = directory+'background_type_after_cosmic.png'
bins = np.linspace(0,100,50)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.reco.pfp.trk.len'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Track Length')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

with open(directory+"Cosmic_Background_Info.txt", "a") as file:
	file.write("\n\n-- After Clear Cosmic  --")
	file.write('\nPurity: ')
	file.write(str(signal_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nCosmics (ratio): ')
	file.write(str(cosmic_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nNeutrino Background (ratio): ')
	file.write(str(nu_bg_df.shape[0]/NC1p_df.shape[0]))
	file.write('\nRatio of selection = protons: ')
	file.write(str(NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0]))
	file.write('\nAbsolute # signal: ')
	file.write(str(signal_df.shape[0]))
	file.write('\nAbsolute # Cosmics: ')
	file.write(str(cosmic_bg_df.shape[0]))
	file.write('\nAbsolute # Neutrino Background: ')
	file.write(str(nu_bg_df.shape[0]))

print('Text file saved as', directory+"Cosmic_Background_Info.txt")
# --------------------------------------

