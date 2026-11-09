"""Coastlines from GSHHG (Wessel and Smith 1996, J. Geophys. Res. 101, 8741-8743), full resolution, level 1
(boundary between land and ocean), for the maps of the paper (figure1_study_area.py, paper_figures.py cases_styled).

Natural Earth 1:10m lacks several small islands of the Hoang Sa (Paracel) and Truong Sa (Spratly) groups, among
them Truong Sa Island itself; GSHHG full resolution has them. File: GSHHS_f_L1.shp of GSHHG 2.3.7 (NOAA NCEI).
"""
import shapefile
from shapely.geometry import box, shape


def land(path, extent, pad=1.0):
    """Land polygons that reach into extent = [lon0, lon1, lat0, lat1], clipped to the extent plus a pad
    (the cut edges then lie outside the map)."""
    x0, x1, y0, y1 = extent
    clip = box(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    out = []
    for s in shapefile.Reader(path).iterShapes(bbox=[x0 - pad, y0 - pad, x1 + pad, y1 + pad]):
        g = shape(s.__geo_interface__)
        if not g.is_valid:
            g = g.buffer(0)
        if g.intersects(clip):
            out.append(g if g.within(clip) else g.intersection(clip))
    return out


def covered(fig, texts, polys, pad_pt=1.0):
    """For each map label (a matplotlib Text on a PlateCarree axes), the number of land polygons under its box,
    halo included (pad_pt points on every side). Island labels must not hide islands: every count should be 0."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    counts = []
    for t in texts:
        e = t.get_window_extent(r).expanded(1.0, 1.0)
        p = pad_pt * fig.dpi / 72
        e = Bbox.from_extents(e.x0 - p, e.y0 - p, e.x1 + p, e.y1 + p)
        (x0, y0), (x1, y1) = t.axes.transData.inverted().transform([[e.x0, e.y0], [e.x1, e.y1]])
        b = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        counts.append((t.get_text().replace("\n", " "), sum(g.intersects(b) for g in polys)))
    return counts
