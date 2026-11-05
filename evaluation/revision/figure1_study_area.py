"""Study-area figure: (a) the study domain on shaded relief, with rivers, regions, the three cases
and an inset of the 0.25 deg input grid over the 0.1 deg target grid; (b) ERA5-Land heavy-rain
frequency over the training years (figure1_data.py). Only the model domain is mapped.

Drawn at the printed size (6.5 in wide), so font sizes are the sizes in the paper.

    python figure1_study_area.py --data $CORRFLOW_OUT/evaluation/figure1 --out <stem>
"""
import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors                    # noqa: E402
import matplotlib.patheffects as pe                    # noqa: E402
import matplotlib.pyplot as plt                        # noqa: E402
import matplotlib.ticker as mticker                    # noqa: E402
import numpy as np                                     # noqa: E402
from cartopy.io import shapereader                     # noqa: E402
from cartopy.mpl.feature_artist import FeatureArtist   # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle   # noqa: E402

LON0, LON1, LAT0, LAT1 = 102.05, 118.05, 5.85, 25.05       # cell edges of the 0.1 deg grid
EXT = [LON0, LON1, LAT0, LAT1]
PC = ccrs.PlateCarree()
CASES = [(21.5, 104.6, "1"), (18.4, 104.9, "2"), (13.7, 109.1, "3")]   # target peaks (case study)
ZOOM = (107.55, 108.55, 15.55, 16.35)                     # Da Nang coast and Annamite foothills
CASE_RED = "#b2182b"
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
SEA = "#e4eef5"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#333333"})


def relief(z):
    """Hypsometric tints on land, light bathymetry at sea, soft hillshade on land."""
    land = z > 0
    tints = mcolors.LinearSegmentedColormap.from_list(
        "land", [(0, "#d3e4bd"), (0.05, "#bfd7a0"), (0.18, "#dcd6a4"), (0.38, "#cdb083"),
                 (0.62, "#a6855f"), (0.85, "#8a6d55"), (1.0, "#efe9e0")])
    rgb = np.where(land[..., None], tints(np.clip(z, 0, 2600) / 2600)[..., :3],
                   np.array(mcolors.to_rgb(SEA)))
    ls = mcolors.LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(np.where(land, z, 0.0), vert_exag=4, dx=1850, dy=1850)
    return np.where(land[..., None], ls.blend_soft_light(rgb, hs[..., None]), rgb), tints


def base(ax, rivers=None):
    ax.set_extent(EXT, crs=PC)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
                   facecolor="none", edgecolor="#3d3d3d", linewidth=0.35, zorder=4)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m"),
                   facecolor="none", edgecolor="#555555", linewidth=0.45, linestyle=(0, (3, 1.6)),
                   zorder=4)
    if rivers is not None:
        ax.add_geometries(rivers["other"], crs=PC, facecolor="none", edgecolor="#7aaed0",
                          linewidth=0.3, zorder=3)
        ax.add_geometries(rivers["major"], crs=PC, facecolor="none", edgecolor="#2b6fa8",
                          linewidth=0.9, zorder=3)
    for a in ax.get_children():                    # outlines as 300-dpi images: small, fast PDF
        if isinstance(a, FeatureArtist):
            a.set_rasterized(True)


def ticks(ax, left=True):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888888", alpha=0.35,
                      linestyle=(0, (4, 3)), zorder=5)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left
    gl.xlocator = mticker.FixedLocator([104, 108, 112, 116])
    gl.ylocator = mticker.FixedLocator([8, 12, 16, 20, 24])
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5, "color": "#333333"}


def cases(ax, size=8.5):
    for la, lo, n in CASES:
        ax.plot(lo, la, "o", ms=size, mfc=CASE_RED, mec="white", mew=0.9, transform=PC, zorder=12)
        ax.text(lo, la, n, color="white", fontsize=6.5, fontweight="bold", ha="center",
                va="center", transform=PC, zorder=13)


def label(ax, lo, la, s, **kw):
    kw = {"fontsize": 6.5, "color": "#222222", "ha": "center", "va": "center", **kw}
    ax.text(lo, la, s, transform=PC, zorder=10, path_effects=HALO, **kw)


def places(ax, da_nang):
    """Regions, cities and island labels, the same on both panels (da_nang: label x, y, ha)."""
    for lo, la, s, rot, fs in [(107.55, 20.15, "Red River\nDelta", 0, 6.5), (105.75, 18.2, "Annamite Range", -58, 6.0),
                               (107.85, 12.75, "Central\nHighlands", 0, 6.5), (105.35, 9.75, "Mekong\nDelta", 0, 6.5)]:
        label(ax, lo, la, s, rotation=rot, style="italic", color="#3a3a3a", fontsize=fs)
    for la, lo, s, (tx, ty, ha) in [(21.03, 105.85, "Hanoi", (106.05, 21.08, "left")),
                                    (16.05, 108.22, "Da Nang", da_nang),
                                    (10.82, 106.63, "Ho Chi Minh City", (106.83, 10.87, "left"))]:
        ax.plot(lo, la, "o", ms=2.6, color="#1a1a1a", mec="white", mew=0.4, transform=PC, zorder=11)
        label(ax, tx, ty, s, ha=ha)
    for la, lo, s in [(15.95, 112.75, "Hoang Sa\n(Paracel Is.)"), (10.9, 115.2, "Truong Sa\n(Spratly Is.)")]:   # English forms, like the city names
        label(ax, lo, la, s, fontsize=5.8, color="#555555")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output stem; .pdf and .png are written")
    args = ap.parse_args()
    d = np.load(os.path.join(args.data, "etopo1_domain.npz"))
    r = np.load(os.path.join(args.data, "heavy_rain_2017_2023.npz"))
    rgb, tints = relief(d["elev"])

    rec = shapereader.Reader(shapereader.natural_earth("10m", "physical", "rivers_lake_centerlines"))
    rivers = {"major": [], "other": []}
    for x in rec.records():
        b = x.geometry.bounds
        if b[2] > LON0 and b[0] < LON1 and b[3] > LAT0 and b[1] < LAT1:
            rivers["major" if x.attributes.get("name_en") in ("Mekong", "Red") else "other"].append(x.geometry)

    # Layout in inches (figure 6.5 x 4.6): two maps of equal size, colorbars underneath
    W, H = 6.5, 4.6
    mw = 2.9
    mh = mw * (LAT1 - LAT0) / (LON1 - LON0)
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]
    ax_a = fig.add_axes(box(0.38, 0.8, mw, mh), projection=PC)
    ax_b = fig.add_axes(box(0.38 + mw + 0.25, 0.8, mw, mh), projection=PC)

    # (a) relief, rivers, regions, cities, cases
    ax_a.imshow(rgb, extent=EXT, origin="upper", transform=PC, interpolation="bilinear", zorder=1)
    base(ax_a, rivers)
    ticks(ax_a)
    places(ax_a, da_nang=(108.3, 16.75, "center"))          # above the zoom box
    label(ax_a, 104.35, 22.55, "Red", style="italic", color="#2b6fa8", rotation=-38)
    label(ax_a, 105.25, 15.2, "Mekong", style="italic", color="#2b6fa8", rotation=-82)
    cases(ax_a)
    ax_a.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], transform=PC,
                             facecolor="none", edgecolor="#111111", linewidth=0.7, zorder=12))

    # inset: native 0.25 deg ERA5 cells over the 0.1 deg ERA5-Land cells
    iw = 0.98
    ih = iw * (ZOOM[3] - ZOOM[2]) / (ZOOM[1] - ZOOM[0])
    x_in = 0.38 + (112.4 - LON0) / (LON1 - LON0) * mw
    y_in = 0.8 + (17.75 - LAT0) / (LAT1 - LAT0) * mh
    ax_i = fig.add_axes(box(x_in, y_in, iw, ih), projection=PC)
    ax_i.imshow(rgb, extent=EXT, origin="upper", transform=PC, interpolation="nearest", zorder=1)
    ax_i.set_extent([ZOOM[0], ZOOM[1], ZOOM[2], ZOOM[3]], crs=PC)
    ax_i.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
                     facecolor="none", edgecolor="#3d3d3d", linewidth=0.5, zorder=4)
    for a in ax_i.get_children():
        if isinstance(a, FeatureArtist):
            a.set_rasterized(True)
    fine = np.arange(ZOOM[0] - 0.05, ZOOM[1] + 0.1, 0.1)       # 0.1 deg cell edges at x.x5
    coarse = np.arange(107.375, ZOOM[1] + 0.25, 0.25)           # 0.25 deg cell edges at k*0.25+0.125
    for v in np.round(np.arange(15.45, ZOOM[3] + 0.1, 0.1), 2):
        ax_i.plot([ZOOM[0], ZOOM[1]], [v, v], color="#4d4d4d", lw=0.3, alpha=0.8, transform=PC, zorder=5)
    for v in np.round(fine, 2):
        ax_i.plot([v, v], [ZOOM[2], ZOOM[3]], color="#4d4d4d", lw=0.3, alpha=0.8, transform=PC, zorder=5)
    for v in np.arange(15.375, ZOOM[3] + 0.25, 0.25):
        ax_i.plot([ZOOM[0], ZOOM[1]], [v, v], color=CASE_RED, lw=0.8, transform=PC, zorder=6)
    for v in coarse:
        ax_i.plot([v, v], [ZOOM[2], ZOOM[3]], color=CASE_RED, lw=0.8, transform=PC, zorder=6)
    ax_a.legend(handles=[Line2D([], [], color=CASE_RED, lw=0.9, label="ERA5 input 0.25°"),
                         Line2D([], [], color="#4d4d4d", lw=0.4, label="ERA5-Land 0.1°")],
                loc="upper left", bbox_to_anchor=((112.4 - LON0) / (LON1 - LON0), (17.7 - LAT0) / (LAT1 - LAT0)),
                fontsize=5.8, frameon=False, handlelength=1.3, borderaxespad=0, labelspacing=0.3)
    for s in ax_i.spines.values():
        s.set_edgecolor("#111111"); s.set_linewidth(0.7)
    for (xa, ya), (xb, yb) in [((ZOOM[1], ZOOM[3]), (0, 1)), ((ZOOM[1], ZOOM[2]), (0, 0))]:
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=ax_a.transData, xyB=(xb, yb),
                                       coordsB=ax_i.transAxes, color="#111111", lw=0.5, zorder=12))

    # (b) heavy-rain frequency (land only)
    ax_b.set_facecolor(SEA)
    v = r["hours_per_year"]
    levels = [0, 25, 50, 75, 100, 125, 150, 200, 250]
    cmap = plt.get_cmap("YlGnBu")
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend="max")
    im = ax_b.pcolormesh(r["lon"], r["lat"], v, transform=PC, cmap=cmap, norm=norm, shading="nearest",
                         rasterized=True, zorder=1)
    base(ax_b)
    ticks(ax_b, left=False)
    places(ax_b, da_nang=(108.42, 16.1, "left"))
    cases(ax_b)

    # titles and colorbars
    for ax, t in [(ax_a, "(a)  Terrain, rivers and the three cases"),
                  (ax_b, "(b)  Heavy hourly rain in ERA5-Land, 2017–2023")]:
        ax.text(0, 1.012, t, transform=ax.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom")
    cax_a = fig.add_axes(box(0.38 + 0.25, 0.4, mw - 0.5, 0.075))
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 2600), cmap=tints)
    cb = fig.colorbar(sm, cax=cax_a, orientation="horizontal", ticks=[0, 500, 1000, 1500, 2000, 2500])
    cb.set_label("Elevation (m)", fontsize=6.8, labelpad=2)
    cax_b = fig.add_axes(box(0.38 + mw + 0.25 + 0.25, 0.4, mw - 0.5, 0.075))
    cb = fig.colorbar(im, cax=cax_b, orientation="horizontal", ticks=levels)
    cb.set_label("Hours per year above $\\tau_{99}$ = 3.31 mm hr$^{-1}$", fontsize=6.8, labelpad=2)
    for c in (cax_a, cax_b):
        c.tick_params(labelsize=6.2, length=2, width=0.5)
        c.xaxis.set_label_position("bottom")
        for s in c.spines.values():
            s.set_linewidth(0.5)

    fig.savefig(args.out + ".pdf", dpi=300)
    fig.savefig(args.out + ".png", dpi=300)
    print(f"{args.out}.pdf + .png; hours/year above tau99 over land: max {np.nanmax(v):.1f}, "
          f"p95 {np.nanpercentile(v, 95):.1f}, median {np.nanmedian(v):.1f}")


if __name__ == "__main__":
    main()
