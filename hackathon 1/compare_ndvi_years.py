import os, json, argparse, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = src.meta.copy()
    # Replace extreme or nodata with nan
    nodata = meta.get("nodata", None)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    # NDVI valid range sanity
    arr = np.where((arr < -1.5) | (arr > 1.5), np.nan, arr)
    return arr, meta


def reproject_match(src_arr, src_meta, ref_meta):
    """Resample src to ref geometry (CRS, transform, size)."""
    dst = np.empty((ref_meta["height"], ref_meta["width"]), dtype="float32")
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        resampling=Resampling.bilinear,
    )
    dst = np.where(np.isfinite(dst), dst, np.nan)
    return dst


def write_tif(path, arr, ref_meta):
    meta = ref_meta.copy()
    meta.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr, 1)


def render_png(path, arr, vmin=-0.4, vmax=0.4, cmap="RdYlGn"):
    plt.figure(figsize=(6, 6))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def diff(a, b):
    # NDVI change (positive = increase in greenness)
    d = (b - a).astype("float32")
    d[~np.isfinite(a) | ~np.isfinite(b)] = np.nan
    return d


def polygons_from_mask(mask, transform, crs):
    feats = []
    for geom, val in shapes(mask.astype("uint8"), mask=mask.astype("uint8"), transform=transform):
        if val != 1:
            continue
        feats.append(shape(geom))
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=feats, crs=crs)
    # dissolve tiny slivers (< 10 pixels approx)
    return gdf


def summarize(arr):
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "valid_pixels": int(np.isfinite(arr).sum())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y2018", required=True, help="path to 2018 NDVI tif")
    ap.add_argument("--y2021", required=True, help="path to 2021 NDVI tif")
    ap.add_argument("--y2024", required=True, help="path to 2024 NDVI tif")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--loss_threshold", type=float, default=-0.15,
                    help="NDVI decrease threshold for hotspot (e.g., -0.15)")
    ap.add_argument("--gain_threshold", type=float, default=0.15,
                    help="NDVI increase threshold for hotspot (e.g., +0.15)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ndvi18, meta18 = read_tif(args.y2018)
    ndvi21, meta21 = read_tif(args.y2021)
    ndvi24, meta24 = read_tif(args.y2024)

    # Align 2021, 2024 to 2018 grid
    ndvi21a = reproject_match(ndvi21, meta21, meta18) if (
                meta21["crs"] != meta18["crs"] or meta21["transform"] != meta18["transform"] or (
        meta21["width"], meta21["height"]) != (meta18["width"], meta18["height"])) else ndvi21
    ndvi24a = reproject_match(ndvi24, meta24, meta18) if (
                meta24["crs"] != meta18["crs"] or meta24["transform"] != meta18["transform"] or (
        meta24["width"], meta24["height"]) != (meta18["width"], meta18["height"])) else ndvi24

    # Diffs
    d18_21 = diff(ndvi18, ndvi21a)
    d21_24 = diff(ndvi21a, ndvi24a)
    d18_24 = diff(ndvi18, ndvi24a)

    # Save GeoTIFFs & PNGs
    write_tif(os.path.join(args.outdir, "ndvi_2018_aligned.tif"), ndvi18, meta18)
    write_tif(os.path.join(args.outdir, "ndvi_2021_aligned.tif"), ndvi21a, meta18)
    write_tif(os.path.join(args.outdir, "ndvi_2024_aligned.tif"), ndvi24a, meta18)
    write_tif(os.path.join(args.outdir, "ndvi_diff_2018_2021.tif"), d18_21, meta18)
    write_tif(os.path.join(args.outdir, "ndvi_diff_2021_2024.tif"), d21_24, meta18)
    write_tif(os.path.join(args.outdir, "ndvi_diff_2018_2024.tif"), d18_24, meta18)

    render_png(os.path.join(args.outdir, "ndvi_diff_2018_2021.png"), d18_21)
    render_png(os.path.join(args.outdir, "ndvi_diff_2021_2024.png"), d21_24)
    render_png(os.path.join(args.outdir, "ndvi_diff_2018_2024.png"), d18_24)

    # Hotspots: loss and gain on 2018→2024
    loss_mask = (d18_24 <= args.loss_threshold) & np.isfinite(d18_24)
    gain_mask = (d18_24 >= args.gain_threshold) & np.isfinite(d18_24)

    loss_gdf = polygons_from_mask(loss_mask, meta18["transform"], meta18["crs"])
    gain_gdf = polygons_from_mask(gain_mask, meta18["transform"], meta18["crs"])

    # area (m²) — only exact if CRS is projected (meter units). If WGS84, area will be in degrees² (use later reprojection for meters).
    if not loss_gdf.empty:
        loss_gdf["area"] = loss_gdf.area
        loss_gdf.to_file(os.path.join(args.outdir, "hotspots_loss.geojson"), driver="GeoJSON")
    if not gain_gdf.empty:
        gain_gdf["area"] = gain_gdf.area
        gain_gdf.to_file(os.path.join(args.outdir, "hotspots_gain.geojson"), driver="GeoJSON")

    # Stats for LLM
    s18 = summarize(ndvi18);
    s21 = summarize(ndvi21a);
    s24 = summarize(ndvi24a)
    sd1824 = summarize(d18_24)

    facts = {
        "time_windows": ["2018→2021", "2021→2024", "2018→2024"],
        "ndvi_means": {"2018": s18["mean"], "2021": s21["mean"], "2024": s24["mean"]},
        "ndvi_change": {"2018_2024_mean": sd1824["mean"], "min": sd1824["min"], "max": sd1824["max"]},
        "hotspots_loss_count": 0 if loss_gdf.empty else int(len(loss_gdf)),
        "hotspots_gain_count": 0 if gain_gdf.empty else int(len(gain_gdf)),
    }
    pd.DataFrame([facts]).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "facts.json"), "w") as f:
        json.dump(facts, f, indent=2)

    print(json.dumps({"ok": True, "outdir": args.outdir, **facts}))


if __name__ == "__main__":
    main()
