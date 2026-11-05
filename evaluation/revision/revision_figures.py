"""Calibration, trade-off and spectra figures and the performance-cost table, from saved scores (CPU only, nothing re-scored).

    python revision_figures.py --out $CORRFLOW_OUT/evaluation/figures --figures calibration tradeoff costtable

calibration  Rank histograms (all land cells; wet cells, ERA5-Land >= 0.1 mm/hr), central-interval coverage on wet cells and
             reliability at 1, 3.31 and 10 mm/hr with the number of cells per forecast probability, on the broad 2025
             sample (1,999 hours, not selected by the target): CorrDiff and CorrFlow seed 0, the range of seeds 0-3 for
             the coverage. From the per-hour counts in scores/test_<method>.npz.
tradeoff     Spread-skill ratio and CRPS against network evaluations per member: CorrFlow Euler 1/5/10/25 (seed 0, and the
             mean of seeds 0-3 dashed) and Heun 5/10 (seed 0), CorrDiff Heun 10/25/50/100, on the 300 random and the 300
             heavy-rain 2025 hours of the steps sweep (tables test_steps*_vs_era5land.csv).
costtable    LaTeX rows of the performance-cost table: latency from timing/timing.csv, scores of the 300 random
             hours from test_steps_vs_era5land.csv (seed 0).
spectra      Radial power spectra of ERA5-Land, Bilinear, the U-Net, CorrDet, CorrDiff and CorrFlow seed 0 (members' spectra
             averaged) on the 1,999 hours of the broad 2025 sample, from the saved fields (results/cache/test,
             preds/deterministic_s1/test, ERA5 0.25° files for Bilinear); same procedure as power_spectra.py (_az_psd). Checks:
             the faster binning equals _az_psd on every field type, and the ERA5-Land curve of the old 2,000-hour sample of
             psd_results.npz is reproduced.
spectra_plot the spectra figure redrawn from psd_broad2025.npz (no recomputing), ERA5-Land as a grey and as a yellow band.

Every drawn number that also exists in a bootstrap table is recomputed here with bootstrap.aggregate and checked against
that table (the run stops on a mismatch). Colors: reference categorical palette, slots 1-3 (validated for all pairs);
marker shape and line style carry identity as well.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.ticker as mticker                                      # noqa: E402
import numpy as np                                                       # noqa: E402

from bootstrap import aggregate                                          # noqa: E402

from paths import EVAL, RESULTS, ZARR                                    # noqa: E402

R = EVAL + "/"
SCORES, TABLES = R + "scores/", R + "tables/"
C_FLOW, C_DIFF, C_DET = "#2a78d6", "#eb6834", "#1baf7a"                  # palette slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "pdf.fonttype": 42, "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
})
TEXTWIDTH = 6.5                                                          # inches (468 pt, elsarticle 3p)


def table(name, subset):
    return {(r["metric"], r["method"]): float(r["value"]) for r in csv.DictReader(open(TABLES + name)) if r["subset"] == subset}


def scores(method):
    z = dict(np.load(SCORES + f"test_{method}.npz"))
    z["_k"] = 16
    return z


def check(name, mine, theirs, tol=1e-5):            # tables keep 6 significant digits
    if abs(mine - theirs) > tol * max(1.0, abs(theirs)):
        raise SystemExit(f"MISMATCH {name}: figure {mine!r} vs table {theirs!r}")
    print(f"  ok {name}: {mine:.6g}")


def panel_title(ax, s, title):                       # "(a) Title" in bold, as in the other figures
    ax.set_title(f"({s}) {title}", loc="left", fontweight="bold")


def calibration(out):
    tab = table("test_sample_vs_era5land.csv", "sample")
    runs = {m: scores(m) for m in ("corrdiff", "corrflow_s0", "corrflow_s1", "corrflow_s2", "corrflow_s3")}
    w = {m: z["tag_sample"].astype(float) for m, z in runs.items()}
    print("calibration: checks against test_sample_vs_era5land.csv")
    agg = {m: aggregate(z, w[m]) for m, z in runs.items()}
    for m in runs:
        for key in ("delta_rank", "delta_rank_wet", "coverage_50_wet", "coverage_80_wet", "coverage_90_wet", "spread_skill"):
            check(f"{m} {key}", agg[m][key], tab[(key, m)])
    hours = int(runs["corrflow_s0"]["tag_sample"].sum())
    for m in ("corrdiff", "corrflow_s0"):                              # dry cells = all cells - wet cells
        tot = (runs[m]["rank_all"] * w[m][:, None]).sum(0)
        dry = tot - (runs[m]["rank_wet"] * w[m][:, None]).sum(0)
        print(f"  {m} dry cells {dry.sum() / tot.sum():.3f} of all; dry-cell rank shares: " + " ".join(f"{x:.3f}" for x in dry / dry.sum()))
    print(f"  broad-sample hours: {hours}")

    fig = plt.figure(figsize=(TEXTWIDTH, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.55, wspace=0.42, left=0.08, right=0.99, top=0.95, bottom=0.08)
    show = (("corrdiff", "CorrDiff", C_DIFF, "o"), ("corrflow_s0", "CorrFlow", C_FLOW, "s"))
    top = gs[0, :].subgridspec(1, 2, wspace=0.25)                      # coverage is reported in a table
    for j, (key, title) in enumerate((("rank_all", "Rank, all land cells"), ("rank_wet", "Rank, wet cells"))):
        ax = fig.add_subplot(top[0, j])
        ranks = np.arange(1, 18)
        for i, (m, lab, col, _) in enumerate(show):
            cnt = (runs[m][key] * w[m][:, None]).sum(0)
            ax.bar(ranks + (i - 0.5) * 0.42, cnt / cnt.sum(), width=0.40, color=col, edgecolor="white", linewidth=0.3, label=lab)
            print(f"  {m} {key}: lowest rank {cnt[0] / cnt.sum():.3f}, highest {cnt[-1] / cnt.sum():.3f}")
        ax.axhline(1 / 17, color=INK2, lw=0.8, ls="--")
        ax.set_xticks([1, 5, 9, 13, 17])
        ax.set_xlabel("Rank among 16 members")
        ax.set_ylabel("Frequency")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel_title(ax, "ab"[j], title)
        if j == 1:
            ax.legend(loc="upper left")
    thr = runs["corrflow_s0"]["reliability_thresholds"]
    axns = []
    for j, ti in enumerate((1, 2, 3)):
        sub = gs[1, j].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax, axn = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        for i, (m, lab, col, mk) in enumerate(show):
            rel = (runs[m]["reliability"] * w[m][:, None, None, None]).sum(0)[ti]
            n, hits = rel[:, 0], rel[:, 1]
            p = np.arange(17) / 16
            ok = n >= 100                                                  # fewer cells: too noisy to plot
            freq = np.where(ok, hits / np.maximum(n, 1), np.nan)            # lines break at the skipped bins
            ax.plot(p, freq, color=col, marker=mk, ms=3, lw=1.0, label=lab)
            axn.bar(p + (i - 0.5) * 0.026, n, width=0.025, color=col, edgecolor="none")
            print(f"  reliability {thr[ti]:.2f} mm/hr {m}: share at p=0 {n[0] / n.sum():.4f}, p=1 {n[-1] / n.sum():.5f}, "
                  f"between {n[1:-1].sum() / n.sum():.5f}; bins with < 100 cells {int((~ok).sum())}")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("ERA5-Land frequency")
        ax.set_xticklabels([])
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        axn.set_yscale("log")
        axn.set_xlim(-0.03, 1.03)
        axn.set_xlabel("Forecast probability")
        axn.set_ylabel("Cells", fontsize=6.5)
        axns.append(axn)
        panel_title(ax, "cde"[j], f"Reliability, {thr[ti]:.3g} mm hr$^{{-1}}$")
    top = 10 ** np.ceil(np.log10(max(a.get_ylim()[1] for a in axns)))
    for a in axns:                                                         # same scale and ticks under (c)-(e)
        a.set_ylim(10, top)
        a.yaxis.set_major_locator(mticker.FixedLocator([1e3, 1e5, 1e7]))
        a.yaxis.set_minor_locator(mticker.NullLocator())
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"calibration.{ext}"))
    plt.close(fig)
    print(f"  wrote {out}/calibration.pdf")


STEPS = {  # label: (method suffix, network evaluations per member)
    "flow_euler": [("corrflow_s0_euler1", 1), ("corrflow_s0_euler5", 5), ("corrflow_s0_euler10", 10), ("corrflow_s0_euler25", 25)],
    "flow_mean": [("corrflow_euler1", 1), ("corrflow_euler5", 5), ("corrflow_euler10", 10), ("corrflow_euler25", 25)],
    "flow_heun": [("corrflow_s0_heun5", 9), ("corrflow_s0_heun10", 19)],
    "diff": [("corrdiff_heun10", 19), ("corrdiff_heun25", 49), ("corrdiff_heun50", 99), ("corrdiff_heun100", 199)],
}


def tradeoff(out):
    tabs = {"sample": (table("test_steps_vs_era5land.csv", "sample"), "steps_", "300 random hours"),
            "heavy": (table("test_steps_heavy_vs_era5land.csv", "sample"), "steps_heavy_", "300 heavy-rain hours")}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.3), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.93, bottom=0.11, hspace=0.18, wspace=0.32)
    style = {"flow_euler": dict(color=C_FLOW, marker="s", ls="-", label="CorrFlow, Euler, seed 0"),
             "flow_mean": dict(color=C_FLOW, marker="s", ls="--", mfc="white", label="CorrFlow, Euler, mean of seeds 0–3"),
             "flow_heun": dict(color=C_FLOW, marker="^", ls="none", mfc="white", label="CorrFlow, Heun, seed 0"),
             "diff": dict(color=C_DIFF, marker="o", ls="-", label="CorrDiff, Heun")}
    print("tradeoff: values drawn (MAE for reference)")
    for j, sub in enumerate(("sample", "heavy")):
        tab, pre, title = tabs[sub]
        for i, (metric, ylab) in enumerate((("spread_skill", "Spread-skill ratio"), ("CRPS", "CRPS (mm hr$^{-1}$)"))):
            ax = axes[i, j]
            if metric == "spread_skill":
                ax.axhline(1, color=INK2, lw=0.8, ls=":")
            for key, pts in STEPS.items():
                x = [n for _, n in pts]
                y = [tab[(metric, pre + m)] for m, _ in pts]
                ax.plot(x, y, lw=1.1, ms=3.6, **style[key])
                if i == 0:
                    print(f"  [{sub}] {key}: " + ", ".join(f"nfe {n}: SSR {tab[('spread_skill', pre + m)]:.3f} CRPS {tab[('CRPS', pre + m)]:.4f} "
                                                         f"MAE {tab[('MAE', pre + m)]:.4f}" for m, n in pts))
                if i == 0 and key in ("flow_euler", "diff"):
                    steps = [m.rsplit("euler", 1)[-1] if "euler" in m else m.rsplit("heun", 1)[-1] for m, _ in pts]
                    for xx, yy, s in zip(x, y, steps):
                        ax.annotate(s, (xx, yy), textcoords="offset points", xytext=(0, -9 if key == "flow_euler" else 5),
                                    ha="center", fontsize=6, color=INK2)
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 10, 30, 100, 200])
            ax.set_xticklabels(["1", "3", "10", "30", "100", "200"])
            ax.set_ylabel(ylab)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel("Network evaluations per member")
            panel_title(ax, "abcd"[2 * i + j], title)
    axes[0, 0].set_ylim(0, 1.2)
    axes[0, 1].set_ylim(0, 1.2)
    axes[0, 0].legend(loc="lower right", fontsize=6.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"tradeoff.{ext}"))
    plt.close(fig)
    print(f"  wrote {out}/tradeoff.pdf")


def costtable(out):
    tim = {r["config"]: r for r in csv.DictReader(open(R + "timing/timing.csv"))}
    tab = table("test_steps_vs_era5land.csv", "sample")
    base = float(tim["corrflow_euler1_k16"]["mean_s"])
    rows = [("U-Net", "--", "--", "unet", None), ("CorrDet", "--", "--", "deterministic", None),
            ("CorrFlow", "Euler", "1", "corrflow_euler1_k1", None),
            ("CorrFlow", "Euler", "1", "corrflow_euler1_k16", "corrflow_s0_euler1"),
            ("CorrFlow", "Euler", "5", "corrflow_euler5_k16", "corrflow_s0_euler5"),
            ("CorrFlow", "Euler", "10", "corrflow_euler10_k16", "corrflow_s0_euler10"),
            ("CorrFlow", "Euler", "25", "corrflow_euler25_k16", "corrflow_s0_euler25"),
            ("CorrFlow", "Heun", "5", "corrflow_heun5_k16", "corrflow_s0_heun5"),
            ("CorrFlow", "Heun", "10", "corrflow_heun10_k16", "corrflow_s0_heun10"),
            ("CorrDiff", "Heun", "25", "corrdiff_heun25_k1", None), ("CorrDiff", "Heun", "50", "corrdiff_heun50_k1", None),
            ("CorrDiff", "Heun", "10", "corrdiff_heun10_k16", "corrdiff_heun10"),
            ("CorrDiff", "Heun", "25", "corrdiff_heun25_k16", "corrdiff_heun25"),
            ("CorrDiff", "Heun", "50", "corrdiff_heun50_k16", "corrdiff_heun50"),
            ("CorrDiff", "Heun", "100", "corrdiff_heun100_k16", "corrdiff_heun100")]
    lines = []
    for model, solver, steps, cfg, sc in rows:
        t = tim[cfg]
        mean, sd = float(t["mean_s"]), float(t["sd_s"])
        ratio = f"{mean / base:.1f}" if t["members"] == "16" else "--"
        if sc:
            m = "steps_" + sc
            perf = (f"{tab[('MAE', m)]:.4f} & {tab[('CRPS', m)]:.4f} & {tab[('bias', m)]:+.4f}".replace("+", "$+$").replace("-0.", "$-$0.")
                    + f" & {tab[('spread_skill', m)]:.2f} & {tab[('coverage_80_wet', m)]:.2f}")
        else:
            perf = "-- & -- & -- & -- & --"
        lines.append(f"    {model:<9s} & {solver:<5s} & {steps:>3s} & {t['nfe_per_member']:>3s} & {t['members']:>2s} & "
                     f"{mean:.3f} ({sd:.3f}) & {ratio} & {perf} \\\\")
    s = "\n".join(lines)
    open(os.path.join(out, "costtable_rows.tex"), "w").write(s + "\n")
    print("costtable rows:\n" + s)
    print(f"  timing hours: {tim['corrflow_euler1_k16']['hours']}; max sd, 16 members: "
          f"{max(float(t['sd_s']) for t in tim.values() if t['members'] == '16'):.4f} s")


CACHE = os.path.join(RESULTS, "cache", "test") + "/"
OLD_PSD = os.path.join(RESULTS, "psd_results.npz")


def az_psd_ref(field, dx_km=11.1):
    """_az_psd of power_spectra.py, unchanged (H, W taken from the field)."""
    H, W = field.shape
    fill = float(np.nanmean(field))
    f = np.where(np.isnan(field), fill, field)
    f = f - float(f.mean())
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    f = f * win
    F = np.fft.fftshift(np.fft.fft2(f))
    psd = np.abs(F) ** 2 / float(np.sum(win ** 2))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=dx_km))
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=dx_km))
    FX, FY = np.meshgrid(fx, fy)
    Rr = np.sqrt(FX ** 2 + FY ** 2)
    mf = min(np.abs(fx).max(), np.abs(fy).max())
    bins = np.linspace(0, mf, min(H, W) // 2)
    psd_r = np.array([psd[(Rr >= bins[i]) & (Rr < bins[i + 1])].mean() if ((Rr >= bins[i]) & (Rr < bins[i + 1])).any() else 0.0
                      for i in range(len(bins) - 1)])
    return 0.5 * (bins[:-1] + bins[1:]), psd_r


class RadialPSD:
    """Same result as az_psd_ref, with the radial bins computed once."""

    def __init__(self, H, W, dx_km=11.1):
        fx = np.fft.fftshift(np.fft.fftfreq(W, d=dx_km))
        fy = np.fft.fftshift(np.fft.fftfreq(H, d=dx_km))
        FX, FY = np.meshgrid(fx, fy)
        Rr = np.sqrt(FX ** 2 + FY ** 2)
        mf = min(np.abs(fx).max(), np.abs(fy).max())
        bins = np.linspace(0, mf, min(H, W) // 2)
        idx = np.digitize(Rr, bins) - 1                       # bins[i] <= R < bins[i + 1]
        self.nb = len(bins) - 1
        self.valid = (idx >= 0) & (idx < self.nb)
        self.idx = idx[self.valid]
        self.count = np.bincount(self.idx, minlength=self.nb)
        self.win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
        self.norm = float(np.sum(self.win ** 2))
        self.freq = 0.5 * (bins[:-1] + bins[1:])

    def __call__(self, field):
        fill = float(np.nanmean(field))
        f = np.where(np.isnan(field), fill, field)
        f = (f - float(f.mean())) * self.win
        psd = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2 / self.norm
        tot = np.bincount(self.idx, weights=psd[self.valid], minlength=self.nb)
        return np.where(self.count > 0, tot / np.maximum(self.count, 1), 0.0)


def stratified_sample(idxs, months, n_target, rng):
    """_stratified_sample of power_spectra.py, unchanged."""
    idxs = np.asarray(idxs)
    if len(idxs) <= n_target:
        return np.sort(idxs)
    picked = []
    for m in np.unique(months):
        pool = idxs[months == m]
        quota = max(1, int(round(n_target * len(pool) / len(idxs))))
        quota = min(quota, len(pool))
        picked.extend(rng.choice(pool, size=quota, replace=False).tolist())
    picked = np.array(sorted(set(picked)))
    if len(picked) > n_target:
        picked = np.sort(rng.choice(picked, size=n_target, replace=False))
    return picked


def spectra(out):
    import xarray as xr
    from era5 import era5_bilinear_mm
    ds = xr.open_dataset(ZARR, engine="zarr", chunks=None)
    land = ~np.isnan(ds["tp"].isel(time=0).values)
    H, W = land.shape
    psd = RadialPSD(H, W)
    old = np.load(OLD_PSD)
    # check 1: the old 2,000-hour sample gives the saved ERA5-Land curve
    years, months = ds["time"].dt.year.values, ds["time"].dt.month.values
    test_all = np.where(years == 2025)[0]
    old_idx = stratified_sample(test_all, months[test_all], 2000, np.random.default_rng(1234))
    assert len(old_idx) == int(old["n_timesteps"]), (len(old_idx), int(old["n_timesteps"]))
    acc = np.zeros(psd.nb)
    for zi in old_idx:
        acc += psd(np.where(land, np.expm1(ds["tp"].isel(time=int(zi)).values), np.nan))
    rel = float(np.max(np.abs(acc / len(old_idx) / old["ERA5-Land"] - 1)))
    print(f"spectra: check 1, ERA5-Land on the old {len(old_idx)}-hour sample vs psd_results.npz: max rel. difference {rel:.2e}")
    assert rel < 1e-6
    assert np.allclose(psd.freq, old["freq"], rtol=1e-12)
    # the broad sample, from the saved fields
    z = np.load(SCORES + "test_corrflow_s0.npz")
    sel = z["tag_sample"]
    hours, times = z["zi"][sel], z["time"][sel].astype("datetime64[s]")
    names = ["ERA5-Land", "Bilinear", "U-Net", "CorrDet", "CorrDiff", "CorrFlow"]
    acc = {n: np.zeros(psd.nb) for n in names}
    for j, (zi, t) in enumerate(zip(hours, times)):
        c = np.load(CACHE + f"{zi}.npz")
        det = np.load(R + f"preds/deterministic_s1/test/{zi}.npz")["mm"][0]
        bil = np.where(land, era5_bilinear_mm(t), np.nan)
        fields = {"ERA5-Land": c["truth_mm"], "Bilinear": bil, "U-Net": c["unet_mm"], "CorrDet": det}
        if j == 0:                                            # check 2: faster binning = _az_psd, every field type
            for n, f in list(fields.items()) + [("CorrDiff member", c["cd_mm"][0]), ("CorrFlow member", c["cf_mm"][0])]:
                assert np.allclose(psd(f), az_psd_ref(f)[1], rtol=1e-10, atol=0), n
            assert np.array_equal(np.isnan(det), ~land) and np.array_equal(np.isnan(c["truth_mm"]), ~land)
            print("spectra: check 2, RadialPSD equals _az_psd on all six field types; saved fields carry the land mask")
        for n, f in fields.items():
            acc[n] += psd(f)
        acc["CorrDiff"] += np.mean([psd(m) for m in c["cd_mm"]], axis=0)
        acc["CorrFlow"] += np.mean([psd(m) for m in c["cf_mm"]], axis=0)
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{len(hours)} hours", flush=True)
    mean = {n: acc[n] / len(hours) for n in names}
    np.savez(os.path.join(out, "psd_broad2025.npz"), freq=psd.freq, n_hours=len(hours), **{n.replace("-", "_"): v for n, v in mean.items()})
    f = psd.freq
    hi = f > f.max() / 2                                      # wavelengths of about 22-44 km
    mid = (f > f.max() / 8) & (f <= f.max() / 2)              # about 44-180 km
    for n in names[1:]:
        r = mean[n] / mean["ERA5-Land"]
        print(f"  {n:<9s} power / ERA5-Land: 44-180 km {np.mean(r[mid]):.3f}, 22-44 km {np.mean(r[hi]):.3f}, smallest bin {r[-1]:.3f}")
    print(f"  old curves (2,000 hours) for comparison, 22-44 km: " + ", ".join(
        f"{k} {np.mean(old[k][hi] / old['ERA5-Land'][hi]):.3f}" for k in ("ERA5-Bilinear", "CorrDiff-UNet", "CorrDiff_Heun", "CorrFlow_Euler")))

    draw_spectra(out, f, mean, "spectra", BAND["yellow"])
    print(f"  ({len(hours)} hours)")


BAND = {"grey": "#c9c9c9", "yellow": "#f8dc75"}                          # ERA5-Land band under the model curves


def draw_spectra(out, f, mean, name, band):
    """Spectra figure: ERA5-Land as a wide band under the model curves (and the same band at 1 in the ratio panel)."""
    style = {"Bilinear": dict(color="#9a9893", lw=1.2, ls="--"), "U-Net": dict(color=INK2, lw=1.2, ls="-."),
             "CorrDet": dict(color=C_DET, lw=1.2, ls="-"), "CorrDiff": dict(color=C_DIFF, lw=1.2, ls="-"),
             "CorrFlow": dict(color=C_FLOW, lw=1.2, ls="-")}
    ref = dict(color=band, lw=5, solid_capstyle="butt", zorder=1)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.17, wspace=0.3)
    wl = 1 / f
    ax.loglog(wl, mean["ERA5-Land"], label="ERA5-Land", **ref)
    axr.semilogx(wl, np.ones_like(wl), **ref)
    for n, st in style.items():
        ax.loglog(wl, mean[n], label=n, zorder=2, **st)
        axr.semilogx(wl, mean[n] / mean["ERA5-Land"], zorder=2, **st)
    for a in (ax, axr):
        a.invert_xaxis()
        a.set_xlabel("Wavelength (km)")
        a.grid(True, which="major", color=GRID, lw=0.5)
        a.set_axisbelow(True)
    ax.set_ylabel("Power spectral density")
    axr.set_ylabel("Ratio to ERA5-Land")
    ax.legend(loc="lower left", ncol=2, fontsize=6.3)
    panel_title(ax, "a", "Radial power spectrum")
    panel_title(axr, "b", "Power relative to ERA5-Land")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {out}/{name}.pdf")


def spectra_plot(out):
    """Redraw the spectra figure from psd_broad2025.npz (written by spectra) without recomputing, in both band colors."""
    z = np.load(os.path.join(out, "psd_broad2025.npz"))
    names = ["ERA5-Land", "Bilinear", "U-Net", "CorrDet", "CorrDiff", "CorrFlow"]
    mean, f = {n: z[n.replace("-", "_")] for n in names}, z["freq"]
    hi = f > f.max() / 2                                      # about 22-44 km, as in spectra
    print(f"spectra_plot: {int(z['n_hours'])} hours from psd_broad2025.npz; power / ERA5-Land at 22-44 km: " +
          ", ".join(f"{n} {np.mean(mean[n][hi] / mean['ERA5-Land'][hi]):.3f}" for n in names[1:]))
    for b, col in BAND.items():
        draw_spectra(out, f, mean, f"spectra_band_{b}", col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", nargs="+", choices=["calibration", "tradeoff", "costtable", "spectra", "spectra_plot"],
                    default=["calibration", "tradeoff", "costtable"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in args.figures:
        globals()[f](args.out)


if __name__ == "__main__":
    main()
