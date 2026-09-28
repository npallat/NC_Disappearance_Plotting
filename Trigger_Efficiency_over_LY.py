import uproot
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import glob
import ROOT
from scipy.stats import beta as beta_dist

# *********************************
# Photon prescale applied in ISCalcCorrelated::CalcIonAndScint (default fScintPreScale).
# SimPhotonsLite counts are already thinned by this factor at generation, so divide by it to recover the true (pre-prescale) light yield.
PRESCALE = 3.92e-2

# ********* USER INPUT ************
# don't use - input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_run101*/*/prod*root'
#input_files = '/exp/sbnd/data/users/npallat/Trigger/full_prod_chain/singleProtonCM/*/*detsim*root'
#input_files_2 = '/pnfs/sbnd/scratch/users/npallat/combined_trigger_stats_Neutrino_26/prod*DetSim*'
#input_files_3 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May19_2/*/prod*DetSim*'
#input_files_4 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May20/*/prod*DetSim*'
#input_files = '/pnfs/sbnd/scratch/users/npallat/combined/prod*'
input_files = '/pnfs/sbnd/scratch/users/npallat/Trigger/2d_detsim/*/prod*DetSim*'

#ly_bins = np.linspace(0, 5000 / PRESCALE, 25)
ly_bins = np.linspace(0, 5000, 25)
fig_save = '/exp/sbnd/data/users/npallat/Trigger/trigger_efficiency_over_ly.png'
# *********************************

if isinstance(input_files, str):
	files = glob.glob(input_files)
else:
	files = input_files

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# *********************************

def plot_trigger_efficiency(input_files, fig_save, ly_bins=np.linspace(0, 5000, 50)):

    ly_all  = []
    ly_trig = []

    if isinstance(input_files, str):
        files = glob.glob(input_files)
    else:
        files = input_files

    if not files:
        raise FileNotFoundError("No files found")

    print("Processing files...")

    ROOT.gSystem.Load("libnusimdata_SimulationBase")

    for fpath in files:
        f      = ROOT.TFile.Open(fpath)
        reader = ROOT.TTreeReader("Events", f)
        mcpart = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCParticle))(
                     reader, "simb::MCParticles_largeant__G4.obj")
        simphotons = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.sim.SimPhotonsLite))(
                 reader, "sim::SimPhotonsLites_pdfastsim__G4.obj")

        ly_per_event        = []
        contained_per_event = []

        while reader.Next():
            ly_evt       = -1.0
            ly_evt = 0
            for channel in simphotons.Get():
                for time, count in channel.DetectedPhotons:
                    ly_evt += count
            #ly_evt /= PRESCALE
            is_contained = False

            # End position containment check from G4 MCParticle
            for part in mcpart.Get():
                last = part.NumberTrajectoryPoints() - 1
                ex = part.Vx(last)
                ey = part.Vy(last)
                ez = part.Vz(last)
                is_contained = (abs(ex) < 190 and abs(ey) < 190 and 10 < ez < 450)
                break

            ly_per_event.append(ly_evt)
            contained_per_event.append(is_contained)

        f.Close()

        with uproot.open(fpath) as uf:
            tree = uf["Events"]
            trig_arr = tree[
                "bool_opdaq_triggerEmulation_DetSim./bool_opdaq_triggerEmulation_DetSim.obj"
            ].array(library="np")

        for trig_val, contained, ly_val in zip(trig_arr, contained_per_event, ly_per_event):
            if contained:
                ly_all.append(ly_val)
                if trig_val:
                    ly_trig.append(ly_val)

    ly_all  = np.array(ly_all)
    ly_trig = np.array(ly_trig)

    # --- Trigger efficiency ---
    counts_all,  edges = np.histogram(ly_all,  bins=ly_bins)
    counts_trig, _     = np.histogram(ly_trig, bins=ly_bins)

    trig_eff = np.where(counts_all > 0, counts_trig / counts_all, 0.0)

    alpha   = 1 - 0.683
    ci_low  = np.where(counts_all > 0, beta_dist.ppf(alpha / 2,     counts_trig,     counts_all - counts_trig + 1), 0.0)
    ci_high = np.where(counts_all > 0, beta_dist.ppf(1 - alpha / 2, counts_trig + 1, counts_all - counts_trig    ), 0.0)
    ci_low  = np.where(counts_trig == 0,          0.0, ci_low)
    ci_high = np.where(counts_trig == counts_all, 1.0, ci_high)
    trig_eff_err = [trig_eff - ci_low, ci_high - trig_eff]

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # --- Efficiency plot ---
    fig, ax = plt.subplots()

    ax.errorbar(bin_centers, trig_eff, yerr=trig_eff_err,
                fmt='-o', markersize=3, linewidth=1, color=colors[0], capsize=3, elinewidth=1,
                label='Trigger Efficiency')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(ly_bins[0], ly_bins[-1] * 1.01)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Light Yield [PE]', fontsize=16)
    ax.set_ylabel('Trigger Efficiency', fontsize=16)
    #legend = ax.legend(fontsize=14)
    #ax.add_artist(legend)
    #ax.text(0.46, 0.63, f'SBND Simulation',
    #        transform=ax.transAxes, fontsize=16, color='gray',
    #        verticalalignment='top')

    plt.tight_layout()
    plt.savefig(fig_save, dpi=300, bbox_inches='tight')
    print('Plot saved as', fig_save)

plot_trigger_efficiency(input_files, fig_save, ly_bins=ly_bins)
