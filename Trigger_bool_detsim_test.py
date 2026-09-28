import uproot
import numpy as np
import matplotlib.pyplot as plt
import glob
import ROOT
import argparse

# *********************************
# USAGE: python Trigger_bool_detsim_test.py /path/to/input.root /path/to/output.png
# *********************************

parser = argparse.ArgumentParser(description="Plot of trigger pass/fail from a DetSim, Reco1, or Reco2 file")
parser.add_argument("input_file", help="Path to the input file")
parser.add_argument("output_file", help="Path/filename for the output plot")
args = parser.parse_args()

input_files = args.input_file
fig_save = args.output_file

# *********************************

def plot_trigger_efficiency(input_files, fig_save, bins=np.linspace(0, 0.12, 50)):

    trig_bools = []

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

        with uproot.open(fpath) as uf:
            tree = uf["Events"]
            trig_arr = tree[
                "bool_opdaq_triggerEmulation_DetSim./bool_opdaq_triggerEmulation_DetSim.obj"
            ].array(library="np")
            for b in trig_arr: trig_bools.append(int(b))

    fig, ax = plt.subplots()
    ax.hist(trig_bools, bins=bins, histtype='step')
    ax.set_xlim(0, 2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Trigger Boolean', fontsize=16)
    ax.set_ylabel('Events', fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_save, dpi=300, bbox_inches='tight')
    print('Plot saved as', fig_save)


plot_trigger_efficiency(input_files, fig_save, bins=np.linspace(0, 2, 3))
