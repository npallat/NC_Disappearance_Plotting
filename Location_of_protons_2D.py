import numpy as np
import ROOT
import matplotlib.pyplot as plt
import glob

# ********* USER INPUT ************
input_files = '/pnfs/sbnd/scratch/users/npallat/Trigger/2d_detsim/*/prod*DetSim*'

directory = '/exp/sbnd/data/users/npallat/Trigger/'

# SBND active volume bounds (cm)
X_RANGE = (-200, 200)
Y_RANGE = (-200, 200)
Z_RANGE = (0, 500)

N_BINS = 25  # spatial bins per axis

# Apply the standard containment cut on proton start position
APPLY_CONTAINMENT_CUT = False

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

OUT_PATHS = {
    'XY': directory + 'proton_locations_XY.png',
    'XZ': directory + 'proton_locations_XZ.png',
    'YZ': directory + 'proton_locations_YZ.png',
}

# *********************************

if isinstance(input_files, str):
    files = glob.glob(input_files)
else:
    files = input_files

# ── 1. Collect proton start positions ──────────────────────────────────────
sx_list, sy_list, sz_list = [], [], []

for fpath in files:
    f = ROOT.TFile.Open(fpath)
    reader = ROOT.TTreeReader("Events", f)
    mcpart = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCParticle))(
                 reader, "simb::MCParticles_largeant__G4.obj")

    while reader.Next():
        for part in mcpart.Get():
            if part.PdgCode() == 2212:
                sx, sy, sz = part.Vx(0), part.Vy(0), part.Vz(0)

                if APPLY_CONTAINMENT_CUT:
                    if not (abs(sx) < 190 and abs(sy) < 190 and 10 < sz < 450):
                        break

                sx_list.append(sx)
                sy_list.append(sy)
                sz_list.append(sz)
                break  # take first proton

    f.Close()

sx_all = np.array(sx_list)
sy_all = np.array(sy_list)
sz_all = np.array(sz_list)

print(f'Collected {len(sx_all)} proton start positions')

# ── 2. Plot 2D position histograms ───────────────────────────────────────────
projections = {
    'XY': (sx_all, sy_all, X_RANGE, Y_RANGE, 'X [cm]', 'Y [cm]'),
    'XZ': (sx_all, sz_all, X_RANGE, Z_RANGE, 'X [cm]', 'Z [cm]'),
    'YZ': (sy_all, sz_all, Y_RANGE, Z_RANGE, 'Y [cm]', 'Z [cm]'),
}

for proj_name, (a1, a2, r1, r2, xlabel, ylabel) in projections.items():
    fig, ax = plt.subplots(figsize=(6, 5))

    h, xedges, yedges, im = ax.hist2d(
        a1, a2, bins=N_BINS, range=[r1, r2], cmap='viridis'
    )

    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label('Proton count', fontsize=16)
    #cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=14)
    #ax.set_title(f'Proton Start Position — {proj_name}', fontsize=14)

    # SBND Preliminary watermark
    #ax.text(0.03, 0.97, 'SBND Preliminary', transform=ax.transAxes, fontsize=12, color='gray', fontname='Nimbus Sans', ha='left', va='top')

    plt.tight_layout()

    out_path = OUT_PATHS[proj_name]
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')
