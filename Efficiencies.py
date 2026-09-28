# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl


# ********* USER INPUT ************
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
# *********************************

total = df.shape[0]
prev = 100

with open(directory+"Efficiencies.txt", "w") as file:
	# [CRIT 0] in reconstruction
	df = df.filter(pl.col('rec.slc.reco.npfp').is_not_null())
	file.write("[CRIT 0] in reconstruction: ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total

	# [CRIT 1] npfp <= 2
	df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
	file.write("\n[CRIT 1] npfp <= 2: ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total

	# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
	file.write("\n[CRIT 2] ALL reco objects in slc are proton-like (chi2): ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total

	# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
	file.write("\n[CRIT 3] ALL reco objects in slc are not muon-like (chi2): ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total

	# [CRIT 4] All tracks contained in fiducial volume
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 500))
	df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 500))
	file.write("\n[CRIT 4] All tracks contained in fiducial volume: ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total

	# [CRIT 5] Flash matching
	df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)
	file.write("\n[CRIT 5] Flash matching: ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total
	print(prev)

	# [CRIT 6] Additional CRT cuts
	df = df.filter(pl.col('rec.sbnd_crt_veto.V0') != 1)
	df = df.filter(pl.col('rec.slc.is_clear_cosmic') != 1)
	file.write("\n[CRIT 6] Additional CRT cuts: ")
	file.write(str(100*df.shape[0]/total))
	file.write(", diff: ")
	file.write(str(prev-(100*df.shape[0]/total)))
	prev = 100*df.shape[0]/total
