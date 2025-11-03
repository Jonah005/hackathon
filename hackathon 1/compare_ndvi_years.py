# compare_ndvi_years.py
# HYBRID (deterministic diff + tiny vision narration) — low RAM
# - Uses ONE small model: moondream
# - Renders side-by-side NDVI PNG (2018|2024) + diff PNG for output
# - Sends ONLY the side-by-side image to the model (smaller embedding)
# - Numeric facts are provided as text to ground the narration

from pathlib import Path
import os, sys, io, json, base64, time
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).parent.resolve()
IN_2018_DIR = ROOT / "2018"
IN_2024_DIR = ROOT / "2024"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL = "llava-phi3:3.8b"

GEN_OPTIONS = {
    "num_ctx": 512,        # small but more room than 256
    "num_predict": 220,    # allow longer answer
    "temperature": 0.3
}

OLLAMA_URL_CHAT = "http://127.0.0.1:11434/api/chat"

PERSONA = "Einstein"

NDVI_VMIN, NDVI_VMAX = -0.2, 0.8
DIFF_VMIN, DIFF_VMAX = -0.4, 0.4
LOSS_THR, GAIN_THR   = -0.15, 0.15

def find_ndvi(folder: Path) -> Path:
    for pat in ("*NDVI*.tif", "*NDVI*.tiff", "*ndvi*.tif", "*ndvi*.tiff", "NDVI.tif", "NDVI.tiff"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No NDVI GeoTIFF in {folder} (expected *NDVI*.tif[f]).")

def read_band1(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = src.meta.copy()
    nodata = meta.get("nodata", None)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where((arr < -5) | (arr > 5), np.nan, arr)
    return arr, meta

def align_to_ref(src_arr, src_meta, ref_meta):
    same = (
        src_meta["crs"] == ref_meta["crs"] and
        src_meta["transform"] == ref_meta["transform"] and
        (src_meta["width"], src_meta["height"]) == (ref_meta["width"], ref_meta["height"])
    )
    if same:
        return src_arr
    dst = np.empty((ref_meta["height"], ref_meta["width"]), dtype="float32")
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_meta["transform"], src_crs=src_meta["crs"],
        dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
        resampling=Resampling.bilinear,
    )
    dst[~np.isfinite(dst)] = np.nan
    return dst

def pixel_area_m2(transform, crs) -> float | None:
    try:
        unit = getattr(crs, "axis_info", [None])[0].unit_name.lower() if crs else ""
        if unit and ("metre" in unit or "meter" in unit):
            return abs(transform.a * transform.e)
    except Exception:
        pass
    return None

def side_by_side_png(arrA, arrB, vmin, vmax, cmap, out_path: Path, labels=("2018 NDVI", "2024 NDVI")):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(arrA, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(labels[0]); axes[0].axis("off")
    im1 = axes[1].imshow(arrB, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(labels[1]); axes[1].axis("off")
    fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    # Avoid tight_layout() to prevent the warning on colorbar+subplots
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def to_png(arr, vmin, vmax, cmap, out_path: Path, title=None):
    plt.figure(figsize=(6, 6))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title, fontsize=10)
    plt.axis("off")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

# compare_ndvi_years.py  (only the chat_vision function changed)

# compare_ndvi_years.py  (only the chat_vision function changed)

def chat_vision(model: str, prompt: str, image_b64: str, timeout_s=240) -> str:
    """Send ONE side-by-side image (base64) + structured prompt to Ollama vision model."""
    payload = {
        "model": model,                 # <- use the argument
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64],      # <- use the function parameter
        }],
        "stream": False,
        "options": GEN_OPTIONS,
    }

    try:
        r = requests.post(OLLAMA_URL_CHAT, json=payload, timeout=timeout_s)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach Ollama at http://127.0.0.1:11434. Is it running?")

    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:400]}")

    data = r.json()
    text = (data.get("message") or {}).get("content") or data.get("content") or ""
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty response from model.")
    return text

def main():
    # 1) Inputs
    p2018 = find_ndvi(IN_2018_DIR)
    p2024 = find_ndvi(IN_2024_DIR)

    # 2) Read & align
    ndvi18, meta18 = read_band1(p2018)
    ndvi24, meta24 = read_band1(p2024)
    ndvi24a = align_to_ref(ndvi24, meta24, meta18)

    # 3) Deterministic diff & stats (cheap)
    diff = (ndvi24a - ndvi18).astype("float32")
    diff[~np.isfinite(ndvi18) | ~np.isfinite(ndvi24a)] = np.nan

    s18  = float(np.nanmean(ndvi18))
    s24  = float(np.nanmean(ndvi24a))
    sdiff= float(np.nanmean(diff))

    loss_mask = (diff <= LOSS_THR) & np.isfinite(diff)
    gain_mask = (diff >= GAIN_THR) & np.isfinite(diff)
    loss_count_px = int(loss_mask.sum())
    gain_count_px = int(gain_mask.sum())

    px_area_m2 = pixel_area_m2(meta18["transform"], meta18.get("crs"))
    loss_area_m2 = gain_area_m2 = None
    if px_area_m2:
        loss_area_m2 = float(loss_count_px * px_area_m2)
        gain_area_m2 = float(gain_count_px * px_area_m2)

    # 4) Render outputs
    side_png = OUT_DIR / "ndvi_2018_2024_side_by_side.png"
    diff_png = OUT_DIR / "ndvi_diff_2018_2024.png"
    side_by_side_png(ndvi18, ndvi24a, NDVI_VMIN, NDVI_VMAX, "RdYlGn", side_png)
    to_png(diff, DIFF_VMIN, DIFF_VMAX, "bwr", diff_png, title="NDVI change (2024 - 2018)")

    facts = {
        "ndvi_mean_2018": round(s18, 4) if np.isfinite(s18) else None,
        "ndvi_mean_2024": round(s24, 4) if np.isfinite(s24) else None,
        "ndvi_mean_diff": round(sdiff,4) if np.isfinite(sdiff) else None,
        "loss_threshold": LOSS_THR,
        "gain_threshold": GAIN_THR,
        "loss_pixels": loss_count_px,
        "gain_pixels": gain_count_px,
        "pixel_area_m2": px_area_m2,
        "loss_area_m2": loss_area_m2,
        "gain_area_m2": gain_area_m2,
    }
    (OUT_DIR / "facts_2018_2024.json").write_text(json.dumps(facts, indent=2), "utf-8")

    # 5) LLM narration with ONE image (side-by-side) to save RAM
    prompt = f"""
    You are Mentor Echo speaking like {PERSONA}. You must write a compact, structured environmental brief.

    You are given a side-by-side NDVI image (2018 left, 2024 right) on scale [{NDVI_VMIN}, {NDVI_VMAX}]
    and numeric facts: {json.dumps(facts, separators=(',', ':'))}.

    RESPONSE FORMAT (markdown). Fill every section. 120–180 words total.

    ### Overall trend
    State whether vegetation improved/declined between 2018→2024 and cite the exact means from facts (ndvi_mean_2018, ndvi_mean_2024, ndvi_mean_diff).

    ### Spatial pattern
    Describe where change is visible (north/central/south/east/west) based on the image tones, contrasting the two panels.

    ### Scale of change
    Use loss_pixels and gain_pixels with thresholds to compare how widespread gain vs loss is (no percentages; say “many more gain pixels than loss” if appropriate).

    ### Interpretation
    Give one likely explanation grounded in NDVI (e.g., canopy thickening, cropping intensity, moisture season). Do not output only a list; write 2–3 sentences.

    ### Advisory
    Two sentences with a concrete next step (e.g., field check in the strongest-change quadrant, acquire higher-resolution scenes).

    EXAMPLE STYLE (do NOT copy numbers):
    Overall trend: NDVI rose slightly, indicating mild greening.
    Spatial pattern: gains cluster in the northwest; minor losses along the southeast edge.
    Scale of change: gains outnumber losses by an order of magnitude.
    Interpretation: likely canopy recovery or more intensive cultivation.
    Advisory: prioritize ground truthing in the strongest cluster; monitor seasonal cycles.

    Now produce the brief.
    """.strip()

    side_b64 = img_b64(side_png)
    narration = chat_vision(MODEL, prompt, side_b64, timeout_s=240)

    out_md = OUT_DIR / "summary_2018_2024.md"
    out_md.write_text(narration + "\n", encoding="utf-8")

    print("\n=== Mentor Echo Summary (2018 vs 2024) ===\n")
    print(narration)
    print(f"\nSaved:\n  {out_md}\n  {side_png}\n  {diff_png}\n  {OUT_DIR/'facts_2018_2024.json'}\nModel: {MODEL}")

if __name__ == "__main__":
    main()
