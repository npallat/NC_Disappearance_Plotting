# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl

# ********* USER INPUT ************
#NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/NC1p_curr_selection_no_bary.csv", separator="\t")
#NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/NC1p_short_curr_selection_no_bary.csv", separator="\t")
NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/barycenter/flashTime_cut/'
# *********************************
NC1p_df = NC1p_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in NC1p_df.columns])
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
#NC1p_df = NC1p_df.filter(pl.col('rec.slc.is_clear_cosmic') != 1) 
NC1p_df = NC1p_df.filter(abs(pl.col('rec.slc.barycenterFM.flashTime')) < 5) 

standard_vars = ['signal.NC1p', 'rec.slc.reco.pfp.trk.truth.p.interaction_id']

barycenter_vars = ['rec.slc.barycenterFM.chargeCenter.x',
			'rec.slc.barycenterFM.chargeCenter.y',
			'rec.slc.barycenterFM.chargeCenter.z',
			#'rec.slc.barycenterFM.chargeCenterXLocal',
			#'rec.slc.barycenterFM.chargeTotal',
			'rec.slc.barycenterFM.chargeWidth.x',
			'rec.slc.barycenterFM.chargeWidth.y',
			'rec.slc.barycenterFM.chargeWidth.z',
			#'rec.slc.barycenterFM.deltaT',
			'rec.slc.barycenterFM.deltaY',
			#'rec.slc.barycenterFM.deltaY_Trigger',
			'rec.slc.barycenterFM.deltaZ',
			#'rec.slc.barycenterFM.deltaZ_Trigger',
			#'rec.slc.barycenterFM.flashAsymmetry',
			#'rec.slc.barycenterFM.flashCenter.x',
			'rec.slc.barycenterFM.flashCenter.y',
			'rec.slc.barycenterFM.flashCenter.z',
			#'rec.slc.barycenterFM.flashFirstHit',
			'rec.slc.barycenterFM.flashPEs',
			'rec.slc.barycenterFM.flashTime',
			#'rec.slc.barycenterFM.flashWidth.x',
			'rec.slc.barycenterFM.flashWidth.y',
			'rec.slc.barycenterFM.flashWidth.z',
			#'rec.slc.barycenterFM.overlapY',
			#'rec.slc.barycenterFM.overlapZ',
			'rec.slc.barycenterFM.radius',
			#'rec.slc.barycenterFM.radius_Trigger',
			#'rec.slc.barycenterFM.score',
			#'rec.slc.barycenterFM.chi2']
			]

barycenter_vars_bins = [np.linspace(-350, 350, 28), #'rec.slc.barycenterFM.chargeCenter.x',
			np.linspace(-200, 200, 16), #'rec.slc.barycenterFM.chargeCenter.y',
			np.linspace(-800, 800, 64), #'rec.slc.barycenterFM.chargeCenter.z',
			#'rec.slc.barycenterFM.chargeCenterXLocal',
			#'rec.slc.barycenterFM.chargeTotal',
			np.linspace(0, 30, 30), #'rec.slc.barycenterFM.chargeWidth.x',
			np.linspace(0, 40, 40), #'rec.slc.barycenterFM.chargeWidth.y',
			np.linspace(0, 50, 50), #'rec.slc.barycenterFM.chargeWidth.z',
			#'rec.slc.barycenterFM.deltaT',
			np.linspace(0, 2000, 200), #'rec.slc.barycenterFM.deltaY',
			#'rec.slc.barycenterFM.deltaY_Trigger',
			np.linspace(0, 2000, 200), #'rec.slc.barycenterFM.deltaZ',
			#'rec.slc.barycenterFM.deltaZ_Trigger',
			#'rec.slc.barycenterFM.flashAsymmetry',
			#'rec.slc.barycenterFM.flashCenter.x',
			np.linspace(-200, 200, 16), #np.linspace(-200, 200, 80), #'rec.slc.barycenterFM.flashCenter.y',
			np.linspace(-800, 800, 64), #'rec.slc.barycenterFM.flashCenter.z',
			#'rec.slc.barycenterFM.flashFirstHit',
			np.linspace(0, 35000, 100), #'rec.slc.barycenterFM.flashPEs',
			np.linspace(-500, 1500, 40), #'rec.slc.barycenterFM.flashTime',
			#'rec.slc.barycenterFM.flashWidth.x',
			np.linspace(0, 40, 40), #'rec.slc.barycenterFM.flashWidth.y',
			np.linspace(0, 50, 50), #'rec.slc.barycenterFM.flashWidth.z',
			#'rec.slc.barycenterFM.overlapY',
			#'rec.slc.barycenterFM.overlapZ',
			np.linspace(0, 2000, 200), #'rec.slc.barycenterFM.radius',
			#np.linspace(0, 1, 20), #'rec.slc.barycenterFM.radius_Trigger',
			#np.linspace(0, 1, 50), #'rec.slc.barycenterFM.score',
			#'auto', #'rec.slc.barycenterFM.chi2'
			]

for var in barycenter_vars: standard_vars.append(var)
NC1p_df = NC1p_df.select(standard_vars)

# *********************************
signal_df = NC1p_df.filter(pl.col('signal.NC1p') == 1)
cosmic_bg_df = NC1p_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') < 0)
nu_bg_df = NC1p_df.filter(pl.col('signal.NC1p').is_null()).filter(pl.col('rec.slc.reco.pfp.trk.truth.p.interaction_id') >= 0)

# 1d hists of all barycenter variables
'''
i = 0
for var in barycenter_vars:
	fig_save = directory+var+'_hist.png'
	bins = barycenter_vars_bins[i]
	plt.clf()
	plt.hist(cosmic_bg_df[var], bins=bins, histtype='step', label='Cosmic background', color='tab:blue')
	plt.hist(nu_bg_df[var], bins=bins, histtype='step', label='Neutrino background', color='tab:orange')
	plt.hist(signal_df[var], bins=bins, histtype='step', label='Signal protons', color='tab:green')
	plt.xlabel(var)
	plt.ylabel('count')
	plt.legend()
	plt.savefig(fig_save)
	print('Plot saved as', fig_save)
	i += 1
'''

var = 'rec.slc.barycenterFM.radius'
fig_save = directory+var+'_hist.png'
bins = np.linspace(0, 500, 50)
print(bins)
plt.clf()
plt.hist(cosmic_bg_df[var], bins=bins, histtype='step', label='Cosmic background', color='tab:blue')
plt.hist(nu_bg_df[var], bins=bins, histtype='step', label='Neutrino background', color='tab:orange')
plt.hist(signal_df[var], bins=bins, histtype='step', label='Signal protons', color='tab:green')
plt.xlabel(var)
plt.ylabel('count')
plt.legend()
plt.savefig(fig_save)
print('Plot saved as', fig_save)

'''
fig_save = directory+'2d_cosmic_score_vs_chi2.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,50,50)
plt.clf()
plt.hist2d(cosmic_bg_df['rec.slc.barycenterFM.score'], cosmic_bg_df['rec.slc.barycenterFM.chi2'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Barycenter Flash Match Chi2')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_nu_background_score_vs_chi2.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,50,50)
plt.clf()
plt.hist2d(nu_bg_df['rec.slc.barycenterFM.score'], nu_bg_df['rec.slc.barycenterFM.chi2'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Barycenter Flash Match Chi2')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

fig_save = directory+'2d_signal_score_vs_chi2.png'
bins1 = np.linspace(0,0.2,21)
bins2 = np.linspace(0,50,50)
plt.clf()
plt.hist2d(signal_df['rec.slc.barycenterFM.score'], signal_df['rec.slc.barycenterFM.chi2'], bins=[bins1, bins2])
plt.colorbar()
plt.xlabel('Barycenter Flash Match Score')
plt.ylabel('Barycenter Flash Match Chi2')
plt.savefig(fig_save)
print('Plot saved as', fig_save)

for center in ['rec.slc.barycenterFM.chargeCenter.', 'rec.slc.barycenterFM.flashCenter.']:
	for pair in [['x', 'y'], ['x', 'z'], ['y', 'z']]:
		var1 = center+pair[0]
		var2 = center+pair[1]
		bins1 = np.linspace(-200, 200, 40)
		if (pair[1] == 'z'): bins2 = np.linspace(0, 500, 50)
		else: bins2 = np.linspace(-200, 200, 40)

		fig_save = directory+var1+'_'+str(pair[1])+'_signal_2dhist.png'
		plt.clf()
		plt.hist2d(signal_df[var1], signal_df[var2], bins=[bins1, bins2])
		plt.colorbar()
		plt.xlabel(var1)
		plt.ylabel(var2)
		plt.savefig(fig_save)
		print('Plot saved as', fig_save)

		fig_save = directory+var1+'_'+str(pair[1])+'_cosmic_2dhist.png'
		plt.clf()
		plt.hist2d(cosmic_bg_df[var1], cosmic_bg_df[var2], bins=[bins1, bins2])
		plt.colorbar()
		plt.xlabel(var1)
		plt.ylabel(var2)
		plt.savefig(fig_save)
		print('Plot saved as', fig_save)

		fig_save = directory+var1+'_'+str(pair[1])+'_nu_bg_2dhist.png'
		plt.clf()
		plt.hist2d(nu_bg_df[var1], nu_bg_df[var2], bins=[bins1, bins2])
		plt.colorbar()
		plt.xlabel(var1)
		plt.ylabel(var2)
		plt.savefig(fig_save)
		print('Plot saved as', fig_save)
'''

# --------------------------------------

'''
Notes:

rec.slc.barycenterFM.chargeCenterXLocal :  [nan]
rec.slc.barycenterFM.deltaT :  [-9999.]
rec.slc.barycenterFM.deltaY_Trigger :  [nan]
rec.slc.barycenterFM.deltaZ_Trigger :  [nan]
rec.slc.barycenterFM.flashAsymmetry :  [-9999.]
rec.slc.barycenterFM.flashFirstHit :  [nan]
rec.slc.barycenterFM.flashWidth.x :  [-9999.]
rec.slc.barycenterFM.overlapY :  [nan]
rec.slc.barycenterFM.overlapZ :  [nan]
'''
