# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl

# ********* USER INPUT ************
#df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")
df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")

#directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_Spring26/plots/'
# *********************************

df = df.select('rec.slc.reco.nstub', 'signal.NC1p', 'rec.slc.reco.npfp', 'rec.slc.reco.pfp.trk.truth.p.pdg', 'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton', 'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon', 'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon')
df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in df.columns])
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
#df = df.filter(pl.col('signal.NC1p') == 1)

#df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80))

def plot_with_errors(data, bins, label, color, scale=1.0):
    counts, bin_edges, patches = plt.hist(data, histtype='step', label=label, bins=bins, color=color, weights=np.full(len(data), scale))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    errors = np.sqrt(counts / scale) * scale  # scale sqrt(N) errors accordingly
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, capsize=3)


def make_chi2_plot(df, particle, plane, scale=1, colors=['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']):
    if (particle != 'muon' and particle != 'proton'):
        print("Particle must be muon or proton. Current argument is ", particle)
        return
    if (plane != 0 and plane != 1 and plane != 2):
        print("Plane must be 0, 1, or 2")
        return

    if particle == 'proton': xaxis = 'Proton Chi2 Plane ' + str(plane)
    if particle == 'muon': xaxis = 'Muon Chi2 Plane ' + str(plane)

    fig_save = directory + particle + '_chi2_' + str(plane) + '_cut.png'
    crit = 'rec.slc.reco.pfp.trk.chi2pid.' + str(plane) + '.chi2_' + particle
    bins = np.linspace(0, 500, 80)

    fig, ax = plt.subplots()

    plot_with_errors(df.filter(pl.col('signal.NC1p') == 1)[crit], bins, 'Signal Protons', colors[6])
    plot_with_errors(df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 2212)[crit], bins, 'Protons', colors[2])
    '''
    #plot_with_errors(df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13)[crit], bins, 'Muons (10%)', colors[1], scale=scale)
    plot_with_errors(df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 13)[crit], bins, 'Muons', colors[1]) #, scale=scale)
    plot_with_errors(df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == 22)[crit], bins, 'Photons', colors[0])
    plot_with_errors(df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 11)[crit], bins, 'Electrons', colors[3])
    plot_with_errors(df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 211)[crit], bins, 'Charged Pions', colors[5])
    plot_with_errors(df.filter(abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) == 321)[crit], bins, 'Charged Kaons', colors[6])
    plot_with_errors(df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') == -2147483648)[crit], bins, 'None (5%)', colors[7], scale=0.05)
    plot_with_errors(df.filter((abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) != 11) &
                               (abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) != 13) &
                               (abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) != 321) &
                               (abs(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg')) != 211) &
                               (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 22) &
                               (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != -2147483648) &
                               (pl.col('rec.slc.reco.pfp.trk.truth.p.pdg') != 2212))[crit], bins, 'Other', colors[4])
    '''
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel(xaxis, fontsize=20)
    ax.set_ylabel('Tracks', fontsize=20)
    legend = ax.legend(fontsize=18)
    ax.add_artist(legend)
    #ax.text(0.58, 0.48, f'SBND Preliminary\nPOT: {POT}', transform=ax.transAxes, fontsize=18, color='gray', fontname='Nimbus Sans', verticalalignment='top')
    plt.tight_layout()
    plt.savefig(fig_save, bbox_inches='tight')
    print('Plot saved as', fig_save)


colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# --------------------------------

make_chi2_plot(df, 'proton', 0, 0.1)
make_chi2_plot(df, 'proton', 1, 0.1)
make_chi2_plot(df, 'proton', 2, 0.1)
make_chi2_plot(df, 'muon', 0, 0.1)
make_chi2_plot(df, 'muon', 1, 0.1)
make_chi2_plot(df, 'muon', 2, 0.1)
