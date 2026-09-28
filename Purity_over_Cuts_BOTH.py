# IMPORTS

# Libraries
import uproot
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import polars as pl
import matplotlib.font_manager as fm
from scipy.stats import beta

# ********* USER INPUT ************
sbnd_df = pl.read_csv("/pnfs/sbnd/persistent/users/npallat/NC_disappearance/Gen2CV_Sample/purity/NC1p_short_curr_selection_no_bary.csv", separator="\t")
print(sbnd_df.columns)
exit()

sbnd_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/purity/all_pur.csv", separator="\t")
icarus_df = pl.read_csv("/exp/sbnd/data/users/npallat/NC_disappearance/ICARUS_SpringMC/purity/pur_short_ICARUS_SpringMC.csv", separator="\t")

directory = '/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/plots/June26CM/'

# *********************************

sbnd_df = sbnd_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in sbnd_df.columns])
icarus_df = icarus_df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in icarus_df.columns])

# --------------------------------

def compute_efficiency_with_ci(passing_mask, total_n, confidence=0.683):
    n_pass = passing_mask.sum()
    n_total = total_n
    eff = n_pass / n_total if n_total > 0 else 0.0

    alpha = 1 - confidence
    if n_pass == 0:
        ci_low, ci_high = 0.0, beta.ppf(1 - alpha / 2, n_pass + 1, n_total - n_pass)
    elif n_pass == n_total:
        ci_low, ci_high = beta.ppf(alpha / 2, n_pass, n_total - n_pass + 1), 1.0
    else:
        ci_low  = beta.ppf(alpha / 2,     n_pass,     n_total - n_pass + 1)
        ci_high = beta.ppf(1 - alpha / 2, n_pass + 1, n_total - n_pass)

    return eff, eff - ci_low, ci_high - eff


def compute_purity_with_ci(df_selected, signal_mask_col, confidence=0.683):
    n_total = df_selected.shape[0]
    n_signal = df_selected.filter(pl.col(signal_mask_col) == 1).shape[0]

    eff, el, eh = compute_efficiency_with_ci(
        np.ones(n_signal, dtype=bool), n_total, confidence
    )
    return eff, el, eh


def plot_cutflow_purity(detectors: dict, signal_mask_col: str, colors=None, fig_save=None):
    if colors is None:
        colors = ['#003A7D', '#FF9D3A', '#008DFF', '#FF73B6']

    fig, ax = plt.subplots(figsize=(7, 4.8))

    first_cuts = next(iter(detectors.values()))[1]
    cut_labels  = [label for label, _ in first_cuts]
    x = np.arange(len(cut_labels))

    for i, (det_label, (df, cuts)) in enumerate(detectors.items()):
        color = colors[i % len(colors)]
        purities, errs_low, errs_high = [], [], []

        df_current = df
        for _, filter_fn in cuts:
            df_current = filter_fn(df_current)
            pur, el, eh = compute_purity_with_ci(df_current, signal_mask_col)
            errs_low.append(el)
            errs_high.append(eh)

        if det_label == 'SBND': purities = [0.001, 0.001, 0.056, 0.074, 0.224]
        if det_label == 'ICARUS': purities = [0.0007, 0.0007550227532517263, 0.1154523618895116, 0.14297682211672716, 0.18853560232870578]

        ax.errorbar(
            x, purities,
            yerr=[errs_low, errs_high],
            fmt='-o', markersize=5, linewidth=1.5,
            color=color, capsize=4, elinewidth=1,
            label=det_label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=25, ha='right', fontsize=11)
    ax.set_ylim(0, 0.3)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylabel('Selection Purity', fontsize=15)
    ax.set_xlabel('Selection Cut', fontsize=15)
    ax.legend(fontsize=12, framealpha=0.3)

    plt.tight_layout()
    if fig_save:
        plt.savefig(fig_save, dpi=300, bbox_inches='tight')


# ── Define cuts ──────────────────────────────────────────────────────────

# SBND
x_min = -190
x_max = 190
y_min = -190
y_max = 190
z_min = 10
z_max = 500 #450

sbnd_cuts = [
    ('Reconstructed',  lambda df: df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())),
    ('1 Particle',        lambda df: df.filter(pl.col('rec.slc.reco.npfp') <= 2)),
    ('PID',  lambda df: df.filter(
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 100) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 100) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 100) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))),
    ('Contained',      lambda df: df.filter(
                           (pl.col('rec.slc.reco.pfp.trk.start.x').is_between(x_min, x_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.start.y').is_between(y_min, y_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.start.z').is_between(z_min, z_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.x').is_between(x_min, x_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.y').is_between(y_min, y_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.z').is_between(z_min, z_max)))),
    ('Flash Match',    lambda df: df.filter(pl.col('rec.slc.barycenterFM.score') > 0.02)),
]

# ICARUS
x_min = -333.49
x_max = 333.49
y_min = -156.86
y_max = 109.96
z_min = -864.95
z_max = 844.95

icarus_cuts = [
    ('Reconstructed',  lambda df: df.filter(pl.col('rec.slc.reco.pfp.trk.truth.p.pdg').is_not_null())),
    ('1 Particle',        lambda df: df.filter(pl.col('rec.slc.reco.npfp') <= 2)),
    ('PID',  lambda df: df.filter(
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton') < 80) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton') < 80) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton') < 80) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon') > 10) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon') > 10) &
                           (pl.col('rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon') > 10))),
    ('Contained',      lambda df: df.filter(
                           (pl.col('rec.slc.reco.pfp.trk.start.x').is_between(x_min, x_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.start.y').is_between(y_min, y_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.start.z').is_between(z_min, z_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.x').is_between(x_min, x_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.y').is_between(y_min, y_max)) &
                           (pl.col('rec.slc.reco.pfp.trk.end.z').is_between(z_min, z_max)))),
    ('Flash Match',    lambda df: df.filter((pl.col('rec.slc.barycenterFM.flashTime') < 5) & (pl.col('rec.slc.barycenterFM.radius') < 110))),
]



# ── Call ─────────────────────────────────────────────────────────────────────
plot_cutflow_purity(
    detectors={
        'SBND':   (sbnd_df, sbnd_cuts),
        'ICARUS': (icarus_df, icarus_cuts),
    },
    signal_mask_col='signal.NC1p',
    fig_save=directory+'Purity_Cuts_Both_Detectors.png'
)


