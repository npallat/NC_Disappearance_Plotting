# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl

# ********* USER INPUT ************
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'
save_name = 'purity_eff_chi2_cuts.csv'
# *********************************

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)

all_proton_counts = df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]
all_muon_counts = df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]
all_NC1p_counts = df.filter(pl.col('signal.NC1p') == 1).shape[0]
all_reco_counts = df.shape[0]

# efficiency and purity csv
#define OR filter (uses ALL_reco_df by default)
def OR(mx, df=df):
	return df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < mx) | (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < mx) | (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < mx))

def AND(mx, df=df):
	return df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < mx) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < mx) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < mx))

def AND_leeq(mx, df=df):
	return df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') <= mx) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') <= mx) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') <= mx))

def OR_leeq(mx, df=df):
	return df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') <= mx) | (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') <= mx) | (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') <= mx))

bins = np.linspace(0,500,80)
plt.clf()
# different selection options
eff_NC1p = []
pur_NC1p = []
eff = []
pur = []
eff_m = []
pur_m = []
OR_cuts = [300, 200, 100, 80, 60, 40, 20]
for cut in OR_cuts:
	cut_df = OR(cut, df)
	p_counts = cut_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]
	NC1p_counts = cut_df.filter(pl.col('signal.NC1p') == 1).shape[0]
	m_counts = cut_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]
	eff_NC1p.append(NC1p_counts/all_NC1p_counts)
	pur_NC1p.append(NC1p_counts/all_reco_counts)
	eff.append(p_counts/all_proton_counts)
	pur.append(p_counts/all_reco_counts)
	eff_m.append(m_counts/all_muon_counts)
	pur_m.append(m_counts/all_reco_counts)
AND_cuts = [300, 200, 100, 80, 60, 40, 20]
for cut in AND_cuts:
	cut_df = AND(cut, df)
	p_counts = cut_df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212).shape[0]
	NC1p_counts = cut_df.filter(pl.col('signal.NC1p') == 1).shape[0]
	m_counts = cut_df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13).shape[0]
	eff_NC1p.append(NC1p_counts/all_NC1p_counts)
	pur_NC1p.append(NC1p_counts/all_reco_counts)
	eff.append(p_counts/all_proton_counts)
	pur.append(p_counts/all_reco_counts)
	eff_m.append(m_counts/all_muon_counts)
	pur_m.append(m_counts/all_reco_counts)
# UPDATE MANUALLY
labels = ['< 300 (OR)', '< 200 (OR)', '< 100 (OR)', '< 80 (OR)', '< 60 (OR)', '< 40 (OR)', '< 20 (OR)', '< 300 (AND)', '< 200 (AND)', '< 100 (AND)', '< 80 (AND)', '< 60 (AND)', '< 40 (AND)', '< 20 (AND)']
if len(labels) != len(OR_cuts)+len(AND_cuts): 
	print('length error for csv')
	exit()
# save eff and purity in CSV file
eff_pur_df = pl.DataFrame({
	'proton chi2 < x (OR)': labels,
	'NC1p efficiency': eff_NC1p, 
	'NC1p purity': pur_NC1p, 
	'efficiency': eff,
	'purity': pur,
	'muon efficiency': eff_m,
	'muon purity': pur_m,
}
)
file = open(directory+save_name, 'w')
eff_pur_df.write_csv(directory+save_name, separator="\t")
file.close()
print('Saved '+directory+save_name)

