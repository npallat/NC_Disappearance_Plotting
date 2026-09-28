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
# NC1p_df = pl.read_csv("/exp/sbnd/app/users/npallat/NC_disappearance/csv_files/polars/parallel/gen2/with_bg/NC1p_reco_V5_full.csv", separator="\t")
# NC1p_df = pl.read_csv("/exp/sbnd/app/users/npallat/NC_disappearance/csv_files/polars/parallel/gen2/with_bg_bary/NC1p_reco_V5_full.csv", separator="\t")
NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

#directory = '/exp/sbnd/app/users/npallat/NC_disappearance/plots/gen2/cosmic_background_removal/bary_first/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/cosmic_bg/'
# *********************************

NC1p_df = NC1p_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in NC1p_df.columns])
NC1p_df = NC1p_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 450))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 450))

# *********************************

# print a bunch of information

print('Purity: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/NC1p_df.shape[0])
print('Cosmics (ratio): ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.shape[0])
print('Background Cosmics (ratio): ', NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.filter(pl.col('signal.NC1p').is_null()).shape[0])
print('Total selection: ', NC1p_df.shape[0])
print('Ratio of selection = protons: ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0])
print('Clear cosmics: ', NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') == 1).shape[0])
print('Clear cosmics that are signal: ', NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('CRT veto V0: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V0') == 1).shape[0])
print('CRT veto V0 that are signal: ',NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V0') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('CRT veto V1: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V1') == 1).shape[0])
print('CRT veto V1 that are signal: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V1') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('CRT veto V2: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V2') == 1).shape[0])
print('CRT veto V2 that are signal: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V2') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('CRT veto V3: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V3') == 1).shape[0])
print('CRT veto V3 that are signal: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V3') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('CRT veto V4: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V4') == 1).shape[0])
print('CRT veto V4 that are signal: ', NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V4') == 1).filter(pl.col('signal.NC1p') == 1).shape[0])
print('Baryflash match score: ', NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') < 0.02).shape[0])
print('Baryflash match score that are signal: ', NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') < 0.02).filter(pl.col('signal.NC1p') == 1).shape[0])
print('Cosmics (ratio): ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.shape[0])

# --------------------------------------

signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

# plots 

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

fig_save = directory+'bary_flashmatch_score.png'
bins = np.linspace(0,1,41)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('BaryCenter Flash Match Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'bary_flashmatch_score_closeup.png'
bins = np.linspace(0,0.2,21)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('BaryCenter Flash Match Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'bary_flashmatch_score_closeup_2.png'
bins = np.linspace(0,0.01,21)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('BaryCenter Flash Match Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'bary_flashmatch_signal.png'
bins = np.linspace(0,1,41)
plt.clf()
plt.hist(signal_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('BaryCenter Flash Match Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'bary_flashmatch_signal_closeup.png'
bins = np.linspace(0,0.2,21)
plt.clf()
plt.hist(signal_df['rec.slc.barycenterFM.score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('BaryCenter Flash Match Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)


# --------------------------------------
'''
print("\nSignal before cuts: ", signal_df.shape[0]/NC1p_df.shape[0])
print("Cosmics before cuts: ", cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print("Neutrino background before cuts: ", nu_bg_df.shape[0]/NC1p_df.shape[0])

NC1p_df =  NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02);
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

print("\nSignal after bary cut: ", signal_df.shape[0]/NC1p_df.shape[0])
print("Cosmics after bary cut: ", cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print("Neutrino after bary cut: ", nu_bg_df.shape[0]/NC1p_df.shape[0])

NC1p_df =  NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1);
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

print("\nSignal after crt veto: ", signal_df.shape[0]/NC1p_df.shape[0])
print("Cosmics after crt veto: ", cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print("Neutrino after crt veto: ", nu_bg_df.shape[0]/NC1p_df.shape[0])

NC1p_df =  NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1);
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

print("\nSignal after clear cosmic: ", signal_df.shape[0]/NC1p_df.shape[0])
print("Cosmics after clear cosmic: ", cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print("Neutrino after clear cosmic: ", nu_bg_df.shape[0]/NC1p_df.shape[0])

'''
# --------------------------------------

fig_save = directory+'clear_cosmic.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.is_clear_cosmic'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.is_clear_cosmic'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.is_clear_cosmic'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Clear Cosmic')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'nu_score.png'
bins = np.linspace(0,1,21)
plt.clf()
plt.hist(cosmic_bg_df['rec.slc.nu_score'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.slc.nu_score'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.slc.nu_score'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('Neutrino Score')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'crt_veto_V0.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.sbnd_crt_veto.V0'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.sbnd_crt_veto.V0'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.sbnd_crt_veto.V0'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('CRT Veto (V0): Any CRT activity except bottom tagger')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'crt_veto_V1.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.sbnd_crt_veto.V1'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.sbnd_crt_veto.V1'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.sbnd_crt_veto.V1'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('CRT Veto (V1): Any CRT activity except bottom and both top')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'crt_veto_V2.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.sbnd_crt_veto.V2'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.sbnd_crt_veto.V2'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.sbnd_crt_veto.V2'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('CRT Veto (V2): Any CRT activity except bottom and North taggers')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'crt_veto_V3.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.sbnd_crt_veto.V3'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.sbnd_crt_veto.V3'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.sbnd_crt_veto.V3'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('CRT Veto (V3): Any CRT activity except bottom and North taggers and both top')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'crt_veto_V4.png'
bins = np.linspace(0,2,3)
plt.clf()
plt.hist(cosmic_bg_df['rec.sbnd_crt_veto.V4'], bins=bins, histtype='step', label='cosmic background')
plt.hist(nu_bg_df['rec.sbnd_crt_veto.V4'], bins=bins, histtype='step', label='neutrino background')
plt.hist(signal_df['rec.sbnd_crt_veto.V4'], bins=bins, histtype='step', label='signal protons')
plt.xlabel('CRT Veto (V4): Only South Wall')
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

# --------------------------------------
'''
fig_save = directory+'2d_cosmic_V0_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.sbnd_crt_veto.V0'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V0: Any CRT activity except bottom tagger')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_V0_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.sbnd_crt_veto.V0'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V0: Any CRT activity except bottom tagger')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_V1_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.sbnd_crt_veto.V1'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V1')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_V1_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.sbnd_crt_veto.V1'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V1')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_V2_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.sbnd_crt_veto.V2'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V2')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_V2_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.sbnd_crt_veto.V2'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V2')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_V3_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.sbnd_crt_veto.V3'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V3')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_V3_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.sbnd_crt_veto.V3'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V3')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_V4_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.sbnd_crt_veto.V4'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V4')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_V4_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.sbnd_crt_veto.V4'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('CRT Veto V4')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_clearcosmic_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.slc.is_clear_cosmic'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Clear Cosmic')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_clearcosmic_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.slc.is_clear_cosmic'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Clear Cosmic')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_nuscore_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,1,21)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.slc.nu_score'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Neutrino Score')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_nuscore_baryflash.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,1,21)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.slc.nu_score'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Neutrino Cosmic')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_clearcosmic_V0.png'
bins1 = np.linspace(0,2,3)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.sbnd_crt_veto.V0'], cosmic_bg_df['rec.slc.is_clear_cosmic'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('CRT Veto V0')
plt.ylabel('Clear Cosmic')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_clearcosmic_V0.png'
bins1 = np.linspace(0,2,3)
bins2 = np.linspace(0,2,3)
plt.clf()
plt.hist2d(signal_df['rec.sbnd_crt_veto.V0'], signal_df['rec.slc.is_clear_cosmic'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('CRT Veto V0')
plt.ylabel('Clear Cosmic')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_cosmic_nuscore_V0.png'
bins1 = np.linspace(0,2,3)
bins2 = np.linspace(0,1,21)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.sbnd_crt_veto.V0'], cosmic_bg_df['rec.slc.nu_score'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('CRT Veto V0')
plt.ylabel('Netrino Score')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_nuscore_V0.png'
bins1 = np.linspace(0,2,3)
bins2 = np.linspace(0,1,21)
plt.clf()
plt.hist2d(signal_df['rec.sbnd_crt_veto.V0'], signal_df['rec.slc.nu_score'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('CRT Veto V0')
plt.ylabel('Netrino Score')
plt.savefig(fig_save)
print('Plot saved as', fig_save)
'''
# --------------------------------------

starting_signal = NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]
starting_cosmics = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]

# Barycenter flash match
NC1p_df =  NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02);
print('\nPurity: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/NC1p_df.shape[0])
print('Efficiency: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/starting_signal)
curr_signal = NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]
curr_cosmics = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]
print('Absolute loss (cosmics): ', starting_cosmics-curr_cosmics)
print('Absolute loss (signal): ', starting_signal-curr_signal)
print('Cosmics (ratio): ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.shape[0])
print('Ratio of selection = protons: ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0])

# remake background distribution plot
fig_save = directory+'background_type_after_bary.png'
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
last_signal = curr_signal
last_cosmics = curr_cosmics

# CRT veto V0 cut: veto on any CRT activity minus the CRT bottom tagger
NC1p_df =  NC1p_df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1);
print('\nPurity: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/NC1p_df.shape[0])
print('Efficiency: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/starting_signal)
curr_signal = NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]
curr_cosmics = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]
print('Absolute loss (cosmics): ', last_cosmics-curr_cosmics)
print('Absolute loss (signal): ', last_signal-curr_signal)
print('Cosmics (ratio): ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.shape[0])
print('Ratio of selection = protons: ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0])

last_signal = curr_signal
last_cosmics = curr_cosmics

# Pandora "clear cosmic" removal
NC1p_df =  NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1);
#NC1p_df =  NC1p_df.filter(pl.col('rec.slc.nu_score') > 0.4);
print('\nPurity: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/NC1p_df.shape[0])
print('Efficiency: ', NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]/starting_signal)
curr_signal = NC1p_df.filter(pl.col('signal.NC1p') == 1).shape[0]
curr_cosmics = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]
print('Absolute loss (cosmics): ', last_cosmics-curr_cosmics)
print('Absolute loss (signal): ', last_signal-curr_signal)
print('Cosmics (ratio): ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.shape[0])
print('Background Cosmics (ratio): ', NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1).shape[0]/NC1p_df.filter(pl.col('signal.NC1p').is_null()).shape[0])
print('Ratio of selection = protons: ', NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]/NC1p_df.shape[0])

signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') == -1)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') != -1)

print("\nSignal: ", signal_df.shape[0]/NC1p_df.shape[0])
print("Cosmic: ", cosmic_bg_df.shape[0]/NC1p_df.shape[0])
print("Neutrino: ", nu_bg_df.shape[0]/NC1p_df.shape[0])

# remake background distribution plot
fig_save = directory+'background_type_after_Cuts.png'
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
