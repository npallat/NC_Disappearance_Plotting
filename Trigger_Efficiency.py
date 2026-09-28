import uproot
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import glob
import ROOT
from scipy.stats import beta as beta_dist

# ********* USER INPUT ************
# don't use - input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_run101*/*/prod*root'
#input_files = '/exp/sbnd/data/users/npallat/Trigger/full_prod_chain/singleProtonCM/*/*detsim*root'
#input_files_2 = '/pnfs/sbnd/scratch/users/npallat/combined_trigger_stats_Neutrino_26/prod*DetSim*'
#input_files_3 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May19_2/*/prod*DetSim*'
#input_files_4 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May20/*/prod*DetSim*'
#input_files = '/pnfs/sbnd/scratch/users/npallat/combined/prod*'
#input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_offset_40/prod*DetSim*root'
input_files = '/pnfs/sbnd/scratch/users/npallat/Trigger/2d_detsim/*/prod*DetSim*'
df = pl.read_csv("/pnfs/sbnd/persistent/users/npallat/NC_disappearance/Gen2CV_Sample/efficiency/all_eff.csv", separator="\t")

ke_bins = np.linspace(0, 0.12, 25)
POT = '5.30e+20'
#fig_save = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/PRN/trigger_efficiency.png'
fig_save = '/exp/sbnd/data/users/npallat/Trigger/trigger_efficiency.png'
# *********************************

# truth
truth_df = df

# reco
# [CRIT 1] npfp <= 2
df = df.filter(pl.col('rec.slc.reco.npfp') <= 2)
# [CRIT 2] ALL reco objects in slc are proton-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100))
# [CRIT 3] ALL reco objects in slc are not muon-like (chi2)
df = df.filter((pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) & (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))
# [CRIT 4] All tracks contained in fiducial volume
df = df.filter((pl.col('rec.slc.reco.pfp.trk.start.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.start.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.start.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.start.z') < 500))
df = df.filter((pl.col('rec.slc.reco.pfp.trk.end.x') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.x') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.y') > -190) & (pl.col('rec.slc.reco.pfp.trk.end.y') < 190) & (pl.col('rec.slc.reco.pfp.trk.end.z') > 10) & (pl.col('rec.slc.reco.pfp.trk.end.z') < 500))
# [CRIT 5] Flash matching
df = df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)

# *********************************

if isinstance(input_files, str):
	files = glob.glob(input_files)
else:
	files = input_files


with uproot.open(files[0]) as f:
    tree = f["Events"]
    # Print all branch names
    #for b in tree.keys():
    #    print(b)

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# *********************************

def plot_trigger_efficiency(input_files, fig_save, truth_df=None, reco_df=None, POT="", ke_bins=np.linspace(0, 0.12, 50)):

    proton_mass = 0.938272  # GeV

    ke_all  = []
    ke_trig = []
    
    if isinstance(input_files, str):
        files = glob.glob(input_files)
    else:
        files = input_files
    
    if not files: raise FileNotFoundError("No files found")

    print("Processing files...")

    ROOT.gSystem.Load("libnusimdata_SimulationBase")

    for fpath in files:
        f      = ROOT.TFile.Open(fpath)
        reader = ROOT.TTreeReader("Events", f)
        mctruth = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCTruth))(
                      reader, "simb::MCTruths_generator__SinglesGen.obj")
        mcpart  = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCParticle))(
                      reader, "simb::MCParticles_largeant__G4.obj")

        ke_per_event       = []
        contained_per_event = []

        while reader.Next():
            ke_evt        = -1.0
            is_contained  = False

            # KE from MCTruth
            for truth in mctruth.Get():
                for p in range(truth.NParticles()):
                    part = truth.GetParticle(p)
                    if part.PdgCode() == 2212:
                        ke_evt = part.E() - proton_mass
                        break

            # End position from G4 MCParticle
            for part in mcpart.Get():
                last = part.NumberTrajectoryPoints() - 1
                ex = part.Vx(last)
                ey = part.Vy(last)
                ez = part.Vz(last)
                if (abs(ex) < 190 and abs(ey) < 190 and 10 < ez < 450):
                    is_contained = True
                break

            ke_per_event.append(ke_evt)
            contained_per_event.append(is_contained)

        f.Close()

        with uproot.open(fpath) as uf:
            tree = uf["Events"]
            trig_arr = tree[
                "bool_opdaq_triggerEmulation_DetSim./bool_opdaq_triggerEmulation_DetSim.obj"
            ].array(library="np")

        assert len(ke_per_event) == len(trig_arr), \
            f"Event count mismatch in {fpath}: {len(ke_per_event)} vs {len(trig_arr)}"

        for ke, trig_val, contained in zip(ke_per_event, trig_arr, contained_per_event):
            if ke > 0 and contained:
                ke_all.append(ke)
                if trig_val:
                    ke_trig.append(ke)

    ke_all  = np.array(ke_all)
    ke_trig = np.array(ke_trig)

    print(ke_all)
    print(ke_trig)

    # --- Trigger efficiency ---
    counts_all,  edges = np.histogram(ke_all,  bins=ke_bins)
    counts_trig, _     = np.histogram(ke_trig, bins=ke_bins)

    trig_eff     = np.where(counts_all > 0, counts_trig / counts_all, 0.0)
    alpha = 1 - 0.683
    ci_low  = np.where(counts_all > 0, beta_dist.ppf(alpha / 2,       counts_trig,     counts_all - counts_trig + 1), 0.0)
    ci_high = np.where(counts_all > 0, beta_dist.ppf(1 - alpha / 2,   counts_trig + 1, counts_all - counts_trig    ), 0.0)
    ci_low  = np.where(counts_trig == 0,          0.0, ci_low)
    ci_high = np.where(counts_trig == counts_all, 1.0, ci_high)
    trig_eff_err = [trig_eff - ci_low, ci_high - trig_eff]
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # --- Reco efficiency (optional) ---
    has_reco = (truth_df is not None) and (reco_df is not None)
    if has_reco:
        ke_truth = (truth_df['rec.mc.nu.prim.startE'] - 0.938).to_numpy()
        ke_reco  = (reco_df['rec.mc.nu.prim.startE']  - 0.938).to_numpy()

        counts_truth, _ = np.histogram(ke_truth, bins=ke_bins[4:])
        counts_reco,  _ = np.histogram(ke_reco,  bins=ke_bins[4:])

        reco_eff     = np.where(counts_truth > 0, counts_reco / counts_truth, 0.0)
        ci_low  = np.where(counts_truth > 0, beta_dist.ppf(alpha / 2,       counts_reco,      counts_truth - counts_reco + 1), 0.0)
        ci_high = np.where(counts_truth > 0, beta_dist.ppf(1 - alpha / 2,   counts_reco + 1,  counts_truth - counts_reco    ), 0.0)
        ci_low  = np.where(counts_reco == 0,           0.0, ci_low)
        ci_high = np.where(counts_reco == counts_truth, 1.0, ci_high)
        reco_eff_err = [reco_eff - ci_low, ci_high - reco_eff]


    print(np.count_nonzero(ke_reco >= 0.06)/len(ke_reco))

    # --- Efficiency plot ---
    fig, ax = plt.subplots()

    print(bin_centers, trig_eff)
    print(min(ke_reco))
    print(len([x for x in ke_reco if x > 0.0625])/len(ke_reco)) 

    ax.errorbar(bin_centers, trig_eff, yerr=trig_eff_err,
                #fmt='o-', color=colors[0], capsize=3,
                fmt='-o', markersize=3, linewidth=1, color=colors[1], capsize=3, elinewidth=1,
                label='Trigger Efficiency')

    if has_reco:
        ax.errorbar(bin_centers[4:], reco_eff, yerr=reco_eff_err,
                    #fmt='s-', color=colors[1], capsize=3,
                    fmt='-s', markersize=3, linewidth=1, color=colors[5], capsize=3, elinewidth=1,
                    label='Selection Efficiency')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, 0.121)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('True Proton Kinetic Energy [GeV]', fontsize=16)
    ax.set_ylabel('Efficiency', fontsize=16)
    legend = ax.legend(fontsize=14)
    ax.add_artist(legend)
    #ax.text(0.68, 0.75, f'SBND Preliminary\nPOT: {POT}',
    ax.text(0.46, 0.69, f'SBN Analysis in Progress',
            transform=ax.transAxes, fontsize=16, color='gray',
            verticalalignment='top')
    ax.text(0.46, 0.63, f'SBND Simulation',
            transform=ax.transAxes, fontsize=16, color='gray',
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(fig_save, dpi=300, bbox_inches='tight')
    print('Plot saved as', fig_save)


plot_trigger_efficiency(input_files, fig_save, truth_df, df, POT, ke_bins=ke_bins)
