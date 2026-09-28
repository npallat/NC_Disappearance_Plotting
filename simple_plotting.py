import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta as beta_dist
 
def hist_with_ratio(
    values_a,
    values_b,
    bins=50,
    range=None,
    label_a="A",
    label_b="B",
    color_a="#3A86FF",
    color_b="#FF006E",
    xlabel="x",
    ylabel="Counts",
    title="Histogram with Ratio",
    ratio_ylabel="A / B",
    figsize=(8, 7),
    ratio_ylim=(0, 3),
):
    """
    Plot two histogram lines in the upper panel and their bin-by-bin ratio
    in the lower panel.
 
    Parameters
    ----------
    values_a, values_b : array-like
        Input data for the two distributions.
    bins : int or array-like
        Number of bins or explicit bin edges.
    range : (float, float) or None
        Lower and upper range of bins. If None, uses data extent.
    label_a, label_b : str
        Legend labels for each distribution.
    color_a, color_b : str
        Line/marker colors for each distribution.
    xlabel : str
        X-axis label (shared by both panels).
    ylabel : str
        Y-axis label for the upper panel.
    title : str
        Figure title.
    ratio_ylabel : str
        Y-axis label for the ratio panel.
    figsize : (float, float)
        Figure size in inches.
    ratio_ylim : (float, float)
        Y-axis limits for the ratio panel.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : (ax_main, ax_ratio)  —  the two Axes objects
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
 
    # Shared bin edges so both histograms are comparable
    if range is None:
        lo = min(values_a.min(), values_b.min())
        hi = max(values_a.max(), values_b.max())
        range = (lo, hi)
 
    counts_a, edges = np.histogram(values_a, bins=bins, range=range)
    counts_b, _     = np.histogram(values_b, bins=bins, range=range)
    centers = 0.5 * (edges[:-1] + edges[1:])
 
    # Poisson errors: sqrt(N)
    err_a = np.sqrt(counts_a)
    err_b = np.sqrt(counts_b)

    # Ratio — avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(counts_a > 0, counts_b / counts_a, np.nan)
        confidence = 0.683
        alpha = 1 - confidence
        ci_low = np.where(
            counts_a > 0,
            beta_dist.ppf(alpha / 2, counts_b, counts_a - counts_b + 1),
            np.nan,
        )
        ci_high = np.where(
            counts_a > 0,
            beta_dist.ppf(1 - alpha / 2, counts_b + 1, counts_a - counts_b),
            np.nan,
        )
        ci_low  = np.where(counts_b == 0,       0,   ci_low)
        ci_high = np.where(counts_b == counts_a, 1, ci_high)
        err_ratio = [ratio - ci_low, ci_high - ratio]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[3, 1],   # upper panel 3× taller than ratio panel
        hspace=0.05,            # tight gap between panels
    )
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    
    # ── Upper panel ───────────────────────────────────────────────────────────
    ax_main.step(edges[:-1], counts_a, where="post", color=color_a,
                 label=label_a)
    ax_main.errorbar(centers, counts_a, yerr=err_a, fmt="none",
                     color=color_a, capsize=2)
    ax_main.step(edges[:-1], counts_b, where="post", color=color_b,
                 label=label_b)
    ax_main.errorbar(centers, counts_b, yerr=err_b, fmt="none",
                     color=color_b, capsize=2)
 
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax_main.legend(fontsize=14)
    ax_main.tick_params(labelbottom=False, labelsize=14)   # hide x tick labels on upper panel
    ax_main.set_xlim(range)
    #ax_main.grid(True, linestyle="--", alpha=0.4)
 
    # ── Ratio panel ───────────────────────────────────────────────────────────
    ax_ratio.step(edges[:-1], ratio, where="post",
                  color="#555555")
    ax_ratio.errorbar(centers, ratio, yerr=err_ratio, fmt="none",
                      color="#555555", capsize=2)

    ax_ratio.set_xlabel(xlabel, fontsize=16)
    ax_ratio.set_ylabel(ratio_ylabel, fontsize=16)
    ax_ratio.set_ylim(ratio_ylim)
    #ax_ratio.grid(True, linestyle="--", alpha=0.4)
 
    fig.align_ylabels([ax_main, ax_ratio])
    return fig, (ax_main, ax_ratio)
 
 
def save_hist_with_ratio(
    values_a,
    values_b,
    bins=50,
    range=None,
    label_a="A",
    label_b="B",
    color_a="#3A86FF",
    color_b="#FF006E",
    xlabel="x",
    ylabel="Counts",
    title="Histogram with Ratio",
    ratio_ylabel="B / A",
    figsize=(8, 7),
    ratio_ylim=(0, 3),
    save_path="/exp/sbnd/data/users/npallat/NC_disappearance/Gen2CV_Sample/hist_with_ratio.png"
):
 
    fig, (ax_top, ax_bot) = hist_with_ratio(
        values_a,
        values_b,
        bins,
        range,
        label_a,
        label_b,
        color_a,
        color_b,
        xlabel,
        ylabel,
        title,
        ratio_ylabel,
        figsize,
        ratio_ylim,
    )
 
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print('Saved ', save_path)
