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
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/background_subset.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
# *********************************

df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])

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

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# --------------------------------

signal_df = df.filter(pl.col('signal.NC1p') == 1)
bg_df = df.filter(pl.col('signal.NC1p').is_null())

print(list(bg_df['rec.slc.truth.parent_pdg']))
print(list(bg_df['rec.slc.truth.isnc']))
