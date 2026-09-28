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
#df = pl.read_csv("/pnfs/sbnd/persistent/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")
df = pl.read_csv("test.csv", separator="\t")

fig_save = '/exp/sbnd/data/users/npallat/Trigger/trigger_bool_caf_test.png'

bins = np.linspace(0, 2, 3)
# *********************************

df = df.select('rec.hdr.triggerinfo.passed_trigger')

# --------------------------------

fig, ax = plt.subplots()
ax.hist(df['rec.hdr.triggerinfo.passed_trigger'], bins=bins, histtype='step')
ax.set_xlim(0, 2)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('Trigger Boolean', fontsize=16)
ax.set_ylabel('Events', fontsize=16)
plt.tight_layout()
plt.savefig(fig_save, dpi=300, bbox_inches='tight')
print('Plot saved as', fig_save)
