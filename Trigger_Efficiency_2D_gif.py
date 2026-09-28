import numpy as np
import ROOT
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import beta as beta_dist
from PIL import Image
import io
import glob

# ********* USER INPUT ************
# don't use - input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_run101*/*/prod*root'
#input_files = '/exp/sbnd/data/users/npallat/Trigger/full_prod_chain/singleProtonCM/*/*detsim*root'
#input_files_2 = '/pnfs/sbnd/scratch/users/npallat/combined_trigger_stats_Neutrino_26/prod*DetSim*'
#input_files_3 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May19_2/*/prod*DetSim*'
#input_files_4 = '/pnfs/sbnd/scratch/users/npallat/trigger_eff_May20/*/prod*DetSim*'
#input_files = '/pnfs/sbnd/scratch/users/npallat/combined/prod*'

input_files = '/pnfs/sbnd/scratch/users/npallat/Trigger/2d_detsim/*/prod*DetSim*'

#input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_upgrade_400/*/prod*DetSim*root'
#input_files = '/pnfs/sbnd/scratch/users/npallat/trigger_large_offset_400/*/prod*DetSim*root'

#directory = '/pnfs/sbnd/persistent/users/npallat/NC_disappearance/Gen2CV_Sample/plots/June26CM/'
directory = '/exp/sbnd/data/users/npallat/Trigger/'

# *********************************

proton_mass = 0.938  # GeV

# KE bin edges (GeV) 
ke_edges = np.linspace(0.0, 0.12, 25)

# Spatial bin counts per axis
N_SPATIAL = 5

# SBND active volume bounds (cm)
X_RANGE = (-200, 200)
Y_RANGE = (-200, 200)
Z_RANGE = (0, 500)

# Output gif paths
GIF_PATHS = {
    'XY': directory+'trig_eff_XY.gif',
    'XZ': directory+'trig_eff_XZ.gif',
    'YZ': directory+'trig_eff_YZ.gif',
}

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# *********************************

if isinstance(input_files, str):
	files = glob.glob(input_files)
else:
	files = input_files


with uproot.open(files[0]) as f:
    tree = f["Events"]

# *********************************

# ── 1. Collect per-event data ──────────────────────────────────────────────────
records = []  # list of (sx, sy, sz, ke, triggered)

for fpath in files:
    # --- trigger bools via uproot ---
    with uproot.open(fpath) as uf:
        tree = uf["Events"]
        trig_arr = tree[
            "bool_opdaq_triggerEmulation_DetSim./bool_opdaq_triggerEmulation_DetSim.obj"
        ].array(library="np")

    # --- positions + KE via TTreeReader ---
    f      = ROOT.TFile.Open(fpath)
    reader = ROOT.TTreeReader("Events", f)
    mctruth = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCTruth))(
                  reader, "simb::MCTruths_generator__SinglesGen.obj")
    mcpart  = ROOT.TTreeReaderValue(ROOT.std.vector(ROOT.simb.MCParticle))(
                  reader, "simb::MCParticles_largeant__G4.obj")

    evt_idx = 0
    while reader.Next():
        ke_evt = -1.0
        sx = sy = sz = None

        for truth in mctruth.Get():
            for p in range(truth.NParticles()):
                part = truth.GetParticle(p)
                if part.PdgCode() == 2212:
                    ke_evt = part.E() - proton_mass
                    break

        for part in mcpart.Get():
            if part.PdgCode() == 2212:
                sx = part.Vx(0)
                sy = part.Vy(0)
                sz = part.Vz(0)
                break  # take first proton

        if sx is not None and ke_evt >= 0 and evt_idx < len(trig_arr):
            records.append((sx, sy, sz, ke_evt, bool(trig_arr[evt_idx])))
        evt_idx += 1

    f.Close()

records = np.array(records, dtype=float)
sx_all  = records[:, 0]
sy_all  = records[:, 1]
sz_all  = records[:, 2]
ke_all  = records[:, 3]
tr_all  = records[:, 4].astype(bool)


# ── 2. Build 2D efficiency maps per KE bin ────────────────────────────────────
projections = {
    'XY': (sx_all, sy_all, X_RANGE, Y_RANGE, 'X [cm]', 'Y [cm]'),
    'XZ': (sx_all, sz_all, X_RANGE, Z_RANGE, 'X [cm]', 'Z [cm]'),
    'YZ': (sy_all, sz_all, Y_RANGE, Z_RANGE, 'Y [cm]', 'Z [cm]'),
}

ke_centers = 0.5 * (ke_edges[:-1] + ke_edges[1:])
n_ke = len(ke_centers)


def make_eff_map(a1, a2, r1, r2, ke_mask):
    """Return (eff, n_total) 2D arrays for events passing ke_mask."""
    mask   = ke_mask
    h_trig, xa, ya = np.histogram2d(a1[mask & tr_all], a2[mask & tr_all],
                                     bins=N_SPATIAL, range=[r1, r2])
    h_tot,  _,  _  = np.histogram2d(a1[mask],          a2[mask],
                                     bins=N_SPATIAL, range=[r1, r2])
    eff = np.where(h_tot > 0, h_trig / h_tot, np.nan)
    return eff.T, h_tot.T, xa, ya   # transpose so rows=y, cols=x


# ── 3. Render frames and assemble gifs ───────────────────────────────────────
for proj_name, (a1, a2, r1, r2, xlabel, ylabel) in projections.items():
    frames = []

    for i, (ke_lo, ke_hi) in enumerate(zip(ke_edges[:-1], ke_edges[1:])):
        ke_mask = (ke_all >= ke_lo) & (ke_all < ke_hi)
        n_evt   = ke_mask.sum()

        eff, h_tot, xa, ya = make_eff_map(a1, a2, r1, r2, ke_mask)

        colors_list = ['#ff9d3a', '#4ecb8d', '#36b700']  # Orange to Teal to Green
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('Custom', colors_list)

        fig, ax = plt.subplots(figsize=(6, 5))
        pcm = ax.pcolormesh(xa, ya, eff,
                             vmin=0, vmax=1,
                             cmap=custom_cmap,
                             shading='flat')
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('Trigger Efficiency', fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(
            #f'Trigger Efficiency — {proj_name}\n'
            f'Proton KE: {ke_lo*1000:.1f} - {ke_hi*1000:.1f} MeV',
            #f'(N={n_evt})',
            fontsize=12
        )
        plt.tight_layout()

        # render to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    # save gif
    out_path = GIF_PATHS[proj_name]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,    # ms per frame
        loop=0,
    )
    print(f'Saved {out_path}')


