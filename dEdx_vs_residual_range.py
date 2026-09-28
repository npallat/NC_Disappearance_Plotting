#!/usr/bin/env python3

import argparse
import sys
from collections import Counter

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.interpolate import interp1d

# ============================================================
# USER INPUT
# ============================================================
expectation_file = '/exp/sbnd/data/users/npallat/NC_disappearance/CC_background/dEdxrestemplates.root'
data_file = '/exp/sbnd/data/users/fnicolas/NC1p_cafpyana/CCDebug/analyzeItOutput_R302-8479_SR97-44.root'
directory = '/exp/sbnd/data/users/npallat/NC_disappearance/CC_background/plots/'

colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6', '#C701FF', '#4ECB8D', '#F9E858', '#D83034']

# -----------------------------------

particles = {
    "dedx_range_pro": ("Proton", colors[0]),
    "dedx_range_mu":  ("Muon",   colors[1]),
    #"dedx_range_pi":  ("Pion",   colors[2]),
    #"dedx_range_ka":  ("Kaon",   colors[3]),
}

particles_dual_color = {
    "dedx_range_pro": ("Proton", colors[0], colors[2]),
    "dedx_range_mu":  ("Muon",   colors[1], colors[6]),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_profile(f, key):
    
    prof = f[key]
    values = prof.values()
    edges = prof.axis().edges()
    centers = 0.5 * (edges[:-1] + edges[1:])

    # TProfile bins with no entries come back as 0; drop them so the line
    # doesn't dive to zero at high residual range where statistics run out.
    mask = values != 0
    return centers[mask], values[mask]


def load_track_dedx_vs_range(data_file, tree_path, xbranch, ybranch):

    with uproot.open(data_file) as f:
        tree = f[tree_path]
        x = tree[xbranch].array(library="ak")
        y = tree[ybranch].array(library="ak")
        pdg = tree["TruePrimariesPDG"].array(library="ak")
        run = tree["RunID"].array(library="ak")
        subrun = tree["SubRunID"].array(library="ak")
        event = tree["EventID"].array(library="ak")
 
    if len(x) != len(y):
        sys.exit(f"Mismatched array lengths after flattening: "
                  f"{xbranch}={len(x)}, {ybranch}={len(y)}")
 
    # keep only slices where both branches report exactly one track
    n_tracks_x = ak.num(x, axis=1)
    n_tracks_y = ak.num(y, axis=1)
    single_track = (n_tracks_x == 1) & (n_tracks_y == 1)
 
    x = x[single_track]
    y = y[single_track]
    pdg = pdg[single_track]
    run = run[single_track]
    subrun = subrun[single_track]
    event = event[single_track]

    # drop the now-redundant track axis, leaving one vector of points per slice
    x = x[:, 0]
    y = y[:, 0]
 
    # keep only slices when single track is a proton
    is_proton = []
    for p_list in pdg: 
        if 2212 in p_list: is_proton.append(True)
        else: is_proton.append(False)
    x = x[is_proton]
    y = y[is_proton]
    run = run[is_proton]
    subrun = subrun[is_proton]
    event = event[is_proton]
 
    # keep only tracks whose (run, subrun, event) combination is unique
    run = ak.to_numpy(run)
    subrun = ak.to_numpy(subrun)
    event = ak.to_numpy(event)
 
    keys = list(zip(run, subrun, event))
    counts = Counter(keys)
    keep = np.array([counts[key] == 1 for key in keys], dtype=bool)
 
    x = x[keep]
    y = y[keep]
 
    return x, y


def expectation_dEdx_plot(fig, ax, expectation_file, plotname, length=None):

    with uproot.open(expectation_file) as f:
        keydict = {k.split(";")[0]: k for k in f.keys()}

        missing = [name for name in particles if name not in keydict]
        if missing:
            sys.exit(f"Could not find these TProfiles in the file: {missing}\n"
                      f"Available keys: {sorted(keydict.keys())}")

        for name, (label, color) in particles.items():
            x, y = load_profile(f, keydict[name])
            if length is not None:
                x_tail = length - x
                x = np.concatenate((x, x_tail))
                y = np.concatenate((y, y))
                ax.scatter(x, y, color=color, label=label, marker='.', s=10)
            else:
                ax.scatter(x, y, color=color, label=label, marker='.', s=10)

    return fig, ax


def build_expectation_curve(expectation_file, key):
 
    with uproot.open(expectation_file) as f:
        keydict = {k.split(";")[0]: k for k in f.keys()}
        if key not in keydict:
            sys.exit(f"Could not find '{key}' in {expectation_file}\n"
                      f"Available keys: {sorted(keydict.keys())}")
        prof = f[keydict[key]]
        values = prof.values()
        errors = prof.errors(error_mode="s")
        edges = prof.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
 
        mask = values != 0
        centers, values, errors = centers[mask], values[mask], errors[mask]
 
    order = np.argsort(centers)
    centers, values, errors = centers[order], values[order], errors[order]
 
    # floor zero/empty bin errors at 5% of the expectation value so a chi2
    # term never divides by zero
    errors = np.where(errors > 0, errors, 0.05 * values)
 
    val_func = interp1d(centers, values, bounds_error=False, fill_value=np.nan)
    err_func = interp1d(centers, errors, bounds_error=False, fill_value=np.nan)
    return val_func, err_func
 
 
def compute_chi2(range_side, dedx_side, val_func, err_func, flip_track_length=None):
    if len(range_side) == 0:
        return np.nan
 
    lookup_range = range_side if flip_track_length is None else flip_track_length - range_side
 
    expected = val_func(lookup_range)
    sigma = err_func(lookup_range)
    valid = np.isfinite(expected) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return np.nan
 
    chi2 = np.sum(((dedx_side[valid] - expected[valid]) / sigma[valid]) ** 2)
    ndof = np.sum(valid)
    return chi2 / ndof
 
 
def compute_track_chi2_scores(x_tracks, y_tracks, expectation_file, p):
    mu_val, mu_err = build_expectation_curve(expectation_file, "dedx_range_mu")
    pro_val, pro_err = build_expectation_curve(expectation_file, "dedx_range_pro")
 
    n_tracks = len(x_tracks)
    proton_chi2_start = np.full(n_tracks, np.nan)
    proton_chi2_end = np.full(n_tracks, np.nan)
    muon_chi2_start = np.full(n_tracks, np.nan)
    muon_chi2_end = np.full(n_tracks, np.nan)
 
    for i in range(n_tracks):
        start_r, start_d, end_r, end_d = get_track_side(x_tracks[i], y_tracks[i], p)
 
        proton_chi2_start[i] = compute_chi2(start_r, start_d, pro_val, pro_err)
        proton_chi2_end[i] = compute_chi2(end_r, end_d, pro_val, pro_err)
        muon_chi2_start[i] = compute_chi2(start_r, start_d, mu_val, mu_err)
        muon_chi2_end[i] = compute_chi2(end_r, end_d, mu_val, mu_err)
 
    return {
        "proton_chi2_start": proton_chi2_start,
        "proton_chi2_end": proton_chi2_end,
        "muon_chi2_start": muon_chi2_start,
        "muon_chi2_end": muon_chi2_end,
    }
 

 
# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_selection_histogram(data_file, expectation_file, p, plotname,
                              tree_path="ana/AnaTPCTree", xbranch="PFPTrackRange", ybranch="PFPTrackdEdx"):

    x_tracks, y_tracks = load_track_dedx_vs_range(data_file, tree_path, xbranch, ybranch)
    scores = compute_track_chi2_scores(x_tracks, y_tracks, expectation_file, p)
    
    bins = np.linspace(0, 500, 25)
    score_colors = [
        ("proton_chi2_start", colors[0]),
        ("proton_chi2_end", colors[1]),
        ("muon_chi2_start", colors[2]),
        ("muon_chi2_end", colors[3]),
    ]
        
    clear_idxs = [0, 4, 6, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21, 25]
    possible_idxs = [3, 22, 23]

    clear_handscan = [i in clear_idxs for i in np.arange(0, len(x_tracks))]
    possible_hanscan = [i in possible_idxs for i in np.arange(0, len(x_tracks))]
    no_handscan = [i not in clear_idxs and i not in possible_idxs for i in np.arange(0, len(x_tracks))]

    for name, color in score_colors:
        fig, ax = plt.subplots()
        ax.clear()
        vals = scores[name]

        clear_vals = vals[clear_handscan]
        possible_vals = vals[possible_hanscan]
        not_mp_vals = vals[no_handscan]

        clear_vals = clear_vals[np.isfinite(clear_vals)]
        possible_vals = possible_vals[np.isfinite(possible_vals)]
        not_mp_vals = not_mp_vals[np.isfinite(not_mp_vals)]

        ax.hist(clear_vals, bins=bins, histtype="step", color=colors[0], label='Clear Proton-Muon')
        ax.hist(possible_vals, bins=bins, histtype="step", color=colors[1], label='Possible Proton-Muon')
        ax.hist(not_mp_vals, bins=bins, histtype="step", color=colors[2], label='Not Proton-Muon')
    
        ax.set_xlabel(name.replace("_", " ")+r"$ \chi^2$/ndof", fontsize=16)
        ax.set_ylabel("Tracks", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=13)
                        
        split_plotname = plotname.split(".")
        curr_plotname = "".join([split_plotname[0], '_', str(name), '.', split_plotname[1]])

        fig.tight_layout()  
        fig.savefig(curr_plotname, dpi=300)
        print("Saved plot as ", curr_plotname)
 

def dEdx_plot(expectation_file, plotname, xmax=None, ymax=None, length=None,
                        data_file=None, tree_path="ana/AnaTPCTree",
                        xbranch="PFPTrackRange", ybranch="PFPTrackdEdx",
                        data_label="Track"):

    fig, ax = plt.subplots()

    if data_file is not None:
        dx, dy = load_track_dedx_vs_range(data_file, tree_path, xbranch, ybranch)
        for i, (tx, ty) in enumerate(zip(dx, dy)):

            ax.clear()

            # adjust expectations by length track
            if length is None: 
                fig, ax = expectation_dEdx_plot(fig, ax, expectation_file, plotname, length=np.max(tx))
            else:
                print("Warning: length of track hard-coded as ", length, ". Check if this is intentional.")
                fig, ax = expectation_dEdx_plot(fig, ax, expectation_file, plotname, length=length)

            ax.scatter(tx, ty, color=colors[5], marker='.', s=30, label=data_label)

            ax.set_xlabel("Residual range [cm]", fontsize=16)
            ax.set_ylabel("dE/dx [MeV/cm]", fontsize=16)
            ax.tick_params(labelsize=14)
    
            if xmax is not None: ax.set_xlim(0, xmax)
            else: ax.set_xlim(0, np.max(tx))
            if ymax is not None: ax.set_ylim(0, ymax)
            else: ax.set_ylim(0, 30)

            ax.legend(fontsize=11, loc='upper center')

            # index plotname
            split_plotname = plotname.split(".")
            curr_plotname = "".join([split_plotname[0], '_', str(i), '.', split_plotname[1]])

            fig.tight_layout()
            fig.savefig(curr_plotname, dpi=300)
            print("Saved plot as ", curr_plotname)
    else:
        fig, ax = expectation_dEdx_plot(fig, ax, expectation_file, plotname, length=length)

        ax.set_xlabel("Residual range [cm]", fontsize=16)
        ax.set_ylabel("dE/dx [MeV/cm]", fontsize=16)
        ax.tick_params(labelsize=14)

        if xmax is not None: ax.set_xlim(0, xmax)
        if ymax is not None: ax.set_ylim(0, ymax)
        else: ax.set_ylim(0, 30)

        ax.legend(fontsize=11, loc='upper center')

        fig.tight_layout()
        fig.savefig(plotname, dpi=300)
        print("Saved plot as ", plotname)


def compare_dEdx_data_with_interp(data_file, expectation_file, plotname, select_tracks=[], 
                                  tree_path="ana/AnaTPCTree", xbranch="PFPTrackRange", ybranch="PFPTrackdEdx"):

    fig, ax = plt.subplots()

    all_tracks_x, all_tracks_y = load_track_dedx_vs_range(data_file, tree_path, xbranch, ybranch)

    if len(select_tracks) == 0: select_tracks = np.arange(0, len(select_tracks)-1, 1)

    for i, (tx, ty) in enumerate(zip(all_tracks_x, all_tracks_y)):

        ax.clear()

        # skip non-select tracks
        if i not in select_tracks: continue

        for name, (label, color_1, color_2) in particles_dual_color.items():

            val, err = build_expectation_curve(expectation_file, name)

            start_x = tx[tx < 27]
            start_y = val(start_x)

            track_start_y = ty[0:len(start_y)]
            dist_to_track_start_y = start_y - track_start_y

            ax.scatter(start_x, dist_to_track_start_y, color=color_1, label=label+" vs. Start", marker='.', s=10)

            end_x = tx[tx > max(tx)-27]
            end_y = val(np.abs(end_x - max(tx)))

            track_end_y = ty[-len(end_y):]
            dist_to_track_end_y = end_y - track_end_y

            ax.scatter(end_x, dist_to_track_end_y, color=color_2, label=label+" vs. End", marker='.', s=10)

        ax.set_xlabel("Residual range [cm]", fontsize=16)
        ax.set_ylabel("dE/dx Difference", fontsize=16)
        ax.tick_params(labelsize=14)

        ax.set_xlim(min(tx), max(tx))
        ax.set_ylim(-30, 30)

        ax.legend(fontsize=11, loc='upper center')

        # index plotname
        split_plotname = plotname.split(".")
        curr_plotname = "".join([split_plotname[0], '_track_', str(i), '.', split_plotname[1]])

        fig.tight_layout()
        fig.savefig(curr_plotname, dpi=300)
        print("Saved plot as ", curr_plotname)

 
def summary_compare_dEdx(data_file, expectation_file, plotname, length, bins=np.linspace(-30, 30, 61), select_tracks=[],
                                  tree_path="ana/AnaTPCTree", xbranch="PFPTrackRange", ybranch="PFPTrackdEdx"):

    fig, ax = plt.subplots()

    all_tracks_x, all_tracks_y = load_track_dedx_vs_range(data_file, tree_path, xbranch, ybranch)

    if len(select_tracks) == 0: select_tracks = np.arange(0, len(select_tracks)-1, 1)

    for name, (label, color_1, color_2) in particles_dual_color.items():

        val, err = build_expectation_curve(expectation_file, name)

        avg_start = [] 
        avg_end = [] 

        for i, (tx, ty) in enumerate(zip(all_tracks_x, all_tracks_y)):

            # skip non-select tracks
            if i not in select_tracks: continue

            start_x = tx[tx < length]
            start_y = val(start_x)

            track_start_y = ty[0:len(start_y)]
            dist_to_track_start_y = start_y - track_start_y

            avg_start.append(np.nanmean(dist_to_track_start_y)) 

            end_x = tx[tx > max(tx)-length]
            end_y = val(np.abs(end_x - max(tx)))

            track_end_y = ty[-len(end_y):]
            dist_to_track_end_y = end_y - track_end_y

            avg_end.append(np.nanmean(dist_to_track_end_y)) 

        ax.hist(avg_start, bins=bins, color=color_1, label=label+" vs. Start", histtype='step')
        ax.hist(avg_end, bins=bins, color=color_2, label=label+" vs. End", histtype='step')

    ax.set_xlabel("Average dE/dx Difference", fontsize=16)
    ax.set_ylabel("Number of Tracks", fontsize=16)
    ax.tick_params(labelsize=14)

    ax.set_xlim(np.min(bins), np.max(bins))

    ax.legend(fontsize=11, loc='upper right')

    fig.tight_layout()
    fig.savefig(plotname, dpi=300)
    print("Saved plot as ", plotname)

# ============================================================
# PLOTS
# ============================================================

#dEdx_plot(expectation_file, directory+'dEdx_vs_residual_range_expectation.png')
#dEdx_plot(expectation_file, directory+'dEdx_vs_residual_range_expectation_mirrored_50.png', length=50)
#dEdx_plot(expectation_file, directory+'dEdx_vs_residual_range_expectation_mirrored_80.png', length=80)
#dEdx_plot(expectation_file, directory+'dEdx_vs_residual_range_expectation_with_data.png', data_file=data_file)

'''
p_options = [5, 10, 20, 30]
for p in p_options:
    plotname = directory+'selected_muon__'+str(p)+'pct_track.png'
    plot_selection_histogram(data_file, expectation_file, p, plotname)
'''

#compare_dEdx_data_with_interp(data_file, expectation_file, directory+'diff_dEdx_vs_residual_range_expectation_with_interp.png', select_tracks=[0, 4, 6, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21, 25])
#compare_dEdx_data_with_interp(data_file, expectation_file, directory+'diff_dEdx_vs_residual_range_expectation_with_interp.png', select_tracks=[0, 4, 6, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21, 25])

summary_compare_dEdx(data_file, expectation_file, directory+'summary_compare_dEdx_10cm.png', 10, bins=np.linspace(-10, 10, 21), select_tracks=[0, 4, 6, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21, 25])
summary_compare_dEdx(data_file, expectation_file, directory+'summary_compare_dEdx_20cm.png', 20, bins=np.linspace(-10, 10, 21), select_tracks=[0, 4, 6, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21, 25])

