# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl

# ********* USER INPUT ************
#NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/all_pur.csv", separator="\t")
#NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")
NC1p_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")

#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/'
save_df = 'NC1p_short_curr_selection_no_bary.csv'

expt = 'SBND'
# *********************************
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
	z_max = 500
else:
	'Error with expt name. Exiting...'
	exit()
# *********************************


NC1p_df = NC1p_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in NC1p_df.columns])
NC1p_df = NC1p_df.filter(pl.col('rec.slc.reco.npfp') <= 2)
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
#NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.start.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.start.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.start.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.start.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.start.z') < z_max))
NC1p_df = NC1p_df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > x_min) & (pl.col('rec.slc.reco.pfp.trk.end.x') < x_max) & (pl.col('rec.slc.reco.pfp.trk.end.y') > y_min) & (pl.col('rec.slc.reco.pfp.trk.end.y') < y_max) & (pl.col('rec.slc.reco.pfp.trk.end.z') > z_min) & (pl.col('rec.slc.reco.pfp.trk.end.z') < z_max))
#NC1p_df = NC1p_df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

NC1p_df.write_csv(directory+save_df, separator="\t")
print('Saved ', directory+save_df)

