"""Case-study figures and tables, drawn with the cells of notebooks/results.ipynb.

    python paper_figures.py --out $CORRFLOW_OUT/evaluation/figures --figures cases cases_styled

cases         the case tables and the spread comparison (cells 33-34) of the three 2025 cases, from the cached
              predictions (CorrFlow seed 0, cache_extremes.py). The Bilinear panel and table row are ERA5 interpolated
              in mm/hr (era5.era5_bilinear_mm, as in the scores) instead of the notebook's log-space field.
cases_styled  the three case maps of the paper at print size (6.5 in, the text width), in the style of the study-area
              figure (figure1_study_area.py): same data and panels as the cases figure, rain in a blue-purple
              palette, text and lines as vectors, rain fields and map features rasterized.

The figure code stays in the notebook. Only the needed cells run; the zarr cell stops before its training-quantile
scan (the notebook replaces those quantiles with the extreme_meta.json values anyway); outputs go to --out, so the
results folder is left as it is. No model is loaded and nothing is downloaded.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.normpath(os.path.join(HERE, "..", "..", "notebooks", "results.ipynb"))
# Cells used, with the start of their source (stops if the notebook layout changes)
CELLS = {2: "import os, sys", 3: "# Open zarr", 7: "# Input preparation", 8: "# ERA5 raw 0.25",
         12: "# Load pre-computed metadata", 16: "def _crps_ens", 21: "_LAND  =",
         33: "# 2x3 case figure", 34: "# Spread comparison"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder for the figures and the case tables")
    ap.add_argument("--figures", nargs="+", choices=["cases", "cases_styled"], default=["cases"])
    ap.add_argument("--only", nargs="*", default=None, help="cases_styled: case names to draw (default: all)")
    ap.add_argument("--palette", nargs="+", default=["bupu"], choices=["bupu", "blues", "rain"],
                    help="cases_styled: rain palette(s); each is written as <name>_styled_<palette>")
    ap.add_argument("--gshhg", default=None,
                    help="cases_styled: GSHHS_f_L1.shp, land and coastlines from GSHHG instead of Natural Earth (gshhg.py)")
    args = ap.parse_args()
    cells = json.load(open(NOTEBOOK))["cells"]
    src = {i: "".join(cells[i]["source"]) for i in CELLS}
    for i, head in CELLS.items():
        assert src[i].startswith(head), f"cell {i} starts with {src[i][:40]!r}"
    os.makedirs(args.out, exist_ok=True)

    g = {"__name__": "paper_figures", "FIG_DIR": args.out}
    # cell 2 hard-codes MASTER_PORT 29500, which another job on the node may hold: use a free port
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    old_port = 'os.environ["MASTER_PORT"] = "29500"'
    assert src[2].count(old_port) == 1
    exec(src[2].replace(old_port, f'os.environ["MASTER_PORT"] = "{port}"'), g)
    if "cases" in args.figures or "cases_styled" in args.figures:
        # without the saved metadata, cell 12 would rescan the whole zarr store
        assert os.path.exists(os.path.join(g["OUT_DIR"], "extreme_meta.json"))
        exec(src[3].split("# Training-distribution thresholds")[0], g)
        for i in (7, 8, 12, 16, 21):
            exec(src[i], g)
        g["_times_pd"] = g["pd"].to_datetime(g["_times_zarr"])      # as in cell 14
        from era5 import era5_bilinear_mm
        g["era5_bilinear_mm"] = era5_bilinear_mm
        swaps = [  # Bilinear in mm/hr instead of the log-space field (cell 33)
            ('    tp_c_log = ds_zarr["tp_coarse"].isel(time=int(zi)).values\n'
             '    bl = np.where(land_mask, np.expm1(tp_c_log), np.nan)',
             '    bl = np.where(land_mask, era5_bilinear_mm(ds_zarr["time"].values[int(zi)]), np.nan)'),
            ('(ax_bl,    bl,      "ERA5-Bilinear"),', '(ax_bl,    bl,      "Bilinear"),'),
            ('("ERA5-Bilinear",    bl,      bl[np.newaxis]),', '("Bilinear",         bl,      bl[np.newaxis]),'),
        ]
        for old, new in swaps:
            assert src[33].count(old) == 1, old
            src[33] = src[33].replace(old, new)
        if "cases" in args.figures:
            for i in (33, 34):
                s = src[i].replace("os.path.join(OUT_DIR,", "os.path.join(FIG_DIR,")
                assert "OUT_DIR" not in s, f"cell {i} would still write to the results folder"
                exec(s, g)
        if "cases_styled" in args.figures:
            for pal in args.palette:
                cases_styled(g, args.out, args.only, pal, args.gshhg)


def cases_styled(g, out, only=None, palette="bupu", gshhg=None):
    """Case maps in the design language of the study-area figure (figure1_study_area.py): its extent, coastline, border
    and gridline styles, fonts and halo labels. Rain in a blue-purple sequential palette, not the yellow-green-blue of the
    study-area figure (that panel shows another quantity), with dry land and sea left plain; the same places in every
    panel: the three cities of the study-area figure, named to the right of their dots (over the sea, which is empty in
    panels b-f), and the two island names. The red box is the +-1.5 deg box of the near-peak spread in the case table
    (cell 34 of notebooks/results.ipynb)."""
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd
    from cartopy.mpl.feature_artist import FeatureArtist
    from era5 import era5_bilinear_mm
    plt, ccrs = g["plt"], g["ccrs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "axes.edgecolor": "#333333"})
    PC = ccrs.PlateCarree()
    EXT = [102.05, 118.05, 5.85, 25.05]                       # study-area figure: cell edges of the 0.1 deg grid
    SEA, LAND, RED = "#e7ecef", "#f5f4f0", "#b2182b"          # sea a little greyer than the study-area figure, so light rain stands out
    HALO = [pe.withStroke(linewidth=1.5, foreground="white")]
    bounds = g["RAIN_BOUNDS"]
    n = len(bounds) - 1
    if palette == "rain":                                     # cmocean rain: made for precipitation, light to dark
        import cmocean
        cols = cmocean.cm.rain(np.linspace(0.14, 1.0, n))
    else:                                                     # BuPu (blue to purple) or Blues (one hue, wider steps)
        cols = plt.get_cmap({"bupu": "BuPu", "blues": "Blues"}[palette])(np.linspace(0.22 if palette == "bupu" else 0.25, 1.0, n))
    cmap = mcolors.ListedColormap(cols)
    cmap.set_under((0, 0, 0, 0))                              # below 0.1 mm/hr: the land or sea underneath
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_over({"bupu": "#2a0a2e", "blues": "#041b3d", "rain": "#140a1f"}[palette])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    land = np.asarray(g["land_mask"], bool)
    coast = None
    if gshhg:                                                 # GSHHG has the small islands Natural Earth lacks
        from gshhg import land as gshhg_land
        coast = gshhg_land(gshhg, EXT)
        print(f"GSHHG: {len(coast)} land polygons in the domain")

    def base(ax, left, bottom):
        ax.set_extent(EXT, crs=PC)
        ax.set_facecolor(SEA)
        if coast is None:
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"), facecolor=LAND, edgecolor="none", zorder=0.5)
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m"), facecolor="none",
                           edgecolor="#3d3d3d", linewidth=0.35, zorder=4)
        else:
            ax.add_geometries(coast, crs=PC, facecolor=LAND, edgecolor="none", zorder=0.5)
            ax.add_geometries(coast, crs=PC, facecolor="none", edgecolor="#3d3d3d", linewidth=0.35, zorder=4)
        ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m"), facecolor="none",
                       edgecolor="#555555", linewidth=0.45, linestyle=(0, (3, 1.6)), zorder=4)
        for a in ax.get_children():                           # map features as images, as in the study-area figure
            if isinstance(a, FeatureArtist):
                a.set_rasterized(True)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888888", alpha=0.35, linestyle=(0, (4, 3)), zorder=5)
        gl.top_labels = gl.right_labels = False
        gl.left_labels, gl.bottom_labels = left, bottom
        gl.xlocator = mticker.FixedLocator([104, 108, 112, 116])
        gl.ylocator = mticker.FixedLocator([8, 12, 16, 20, 24])
        gl.xlabel_style = gl.ylabel_style = {"size": 5.8, "color": "#333333"}

    def text(ax, lo, la, s, **kw):
        kw = {"fontsize": 5.8, "color": "#222222", "ha": "left", "va": "center", **kw}
        return ax.text(lo, la, s, transform=PC, zorder=12, path_effects=HALO, **kw)

    def places(ax):                                           # every panel: cities and island names
        # names right of the dots, clear of the +-1.5 deg boxes of all three cases (Hanoi's starts east of the April box)
        for la, lo, name, (tx, ty) in [(21.03, 105.85, "Hanoi", (106.3, 21.03)),
                                       (16.05, 108.22, "Da Nang", (108.45, 16.05)),
                                       (10.82, 106.63, "Ho Chi Minh City", (106.83, 10.87))]:
            ax.plot(lo, la, "o", ms=2.3, color="#1a1a1a", mec="white", mew=0.35, transform=PC, zorder=12)
            text(ax, tx, ty, name)
        # same x: aligned, clear of the boxes; beside the islands, not over them (east of Hoang Sa, north of Truong Sa)
        for la, s in [(16.9, "Hoang Sa\n(Paracel Is.)"), (12.5, "Truong Sa\n(Spratly Is.)")]:
            island_labels.append(text(ax, 115.2, la, s, fontsize=5.2, color="#555555", ha="center"))

    for cname, (cts, split_cs, zoom_spec, peak_ll, fname) in g["CASE_META"].items():
        if only and cname not in only:
            continue
        plat, plon = peak_ll                                  # the star; the spread box is +-1.5 deg around it
        zi = int(np.argmin(np.abs(g["_times_pd"] - pd.Timestamp(cts))))
        truth, unet, cd_ens, cf_ens = g["_get"](split_cs, zi)
        era5_c, era5_lat, era5_lon = g["load_era5_coarse_native"](cts)
        lo2c, la2c = np.meshgrid(era5_lon, era5_lat)
        with np.errstate(all="ignore"):
            cd_mean, cf_mean = np.nanmean(cd_ens, axis=0), np.nanmean(cf_ens, axis=0)
        bil = np.where(land, era5_bilinear_mm(g["ds_zarr"]["time"].values[zi]), np.nan)
        fig = plt.figure(figsize=(6.5, 5.75))
        island_labels = []                                    # filled by places(); checked against GSHHG before saving
        gs = fig.add_gridspec(2, 3, left=0.06, right=0.995, top=0.955, bottom=0.135, wspace=0.05, hspace=0.12)
        panels = [("ERA5 input", lo2c, la2c, era5_c), ("ERA5-Land", None, None, truth), ("Bilinear", None, None, bil),
                  ("U-Net", None, None, unet), ("CorrDiff mean", None, None, cd_mean), ("CorrFlow mean", None, None, cf_mean)]
        for k, (title, lo2, la2, data) in enumerate(panels):
            r, c = divmod(k, 3)
            ax = fig.add_subplot(gs[r, c], projection=PC)
            base(ax, left=(c == 0), bottom=(r == 1))
            x, y = (lo2, la2) if lo2 is not None else (g["lon2d_fine"], g["lat2d_fine"])
            im = ax.pcolormesh(x, y, data, transform=PC, shading="auto", cmap=cmap, norm=norm, rasterized=True, zorder=1)
            ax.add_patch(mpatches.Rectangle((plon - 1.5, plat - 1.5), 3.0, 3.0, transform=PC, facecolor="none",
                                            edgecolor=RED, linewidth=0.7, zorder=10))
            ax.plot(plon, plat, "*", ms=5.5, mfc=RED, mec="white", mew=0.45, transform=PC, zorder=11)
            ax.text(0, 1.015, f"({'abcdef'[k]}) {title}", transform=ax.transAxes, fontsize=7, fontweight="bold",
                    ha="left", va="bottom")
            ax.text(0.04, 0.035, f"peak {np.nanmax(data):.1f} mm hr$^{{-1}}$", transform=ax.transAxes, fontsize=6,
                    color="#222222", zorder=12, path_effects=HALO)
            places(ax)
        cax = fig.add_axes([0.3, 0.07, 0.45, 0.017])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max", ticks=bounds)
        cb.ax.set_xticklabels([f"{v:g}" for v in bounds])
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.set_label("Rainfall (mm hr$^{-1}$)", fontsize=6.8, labelpad=2)
        for ext in ("pdf", "png"):
            if coast is not None and ext == "pdf":
                from gshhg import covered
                print(f"  {fname}: islands under the island labels (must be 0):", covered(fig, island_labels, coast))
            fig.savefig(os.path.join(out, f"{fname}_styled_{palette}.{ext}"), dpi=400)
        plt.close(fig)
        path = os.path.join(out, f"{fname}_styled_{palette}.pdf")
        print(f"  {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
